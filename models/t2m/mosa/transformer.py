import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import clip
import numpy as np
from functools import partial
from utils.tools import cal_performance_naive, eval_decorator, sample_with_top_k_top_p_, gumbel_softmax_with_rng
from models.t2m.mosa.network import Block
from models.t2m.mosa.embed_rope import compute_cis
from utils.tools import lengths_to_mask

class CLIPModelWrapper(nn.Module):
    def __init__(self, clip_version):
        super(CLIPModelWrapper, self).__init__()
        self.load_and_freeze_clip(clip_version)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        print("***clip model loaded***")
        self.clip_model = clip_model

    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)

        # self.clip_model.encode_text
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        # sentence level and word level
        word_mask = text > 0
        # max_len = max(word_mask.sum(dim=-1))
        # x = x[:, :max_len]
        # word_mask = word_mask[:, :max_len]
        return sent_x.float(), x.float(), word_mask


class Transformer(nn.Module):
    def __init__(self, scales, nb_code_st, nb_code_ed, code_dim, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None,
                 **kargs):
        super(Transformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        # 0. hyperparameters
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.pad_id = -1

        self.cond_drop_prob = cond_drop_prob
        
        self.scales = scales
        self.first_t = self.scales[0]
        self.num_stages_minus1 = len(self.scales)-1
        self.rng = torch.Generator('cuda')

        # 1. input (word) embedding
        self.input_process = nn.Linear(self.code_dim, self.latent_dim)
        self.input_emb = nn.Linear(self.clip_dim, self.latent_dim)
        
        # 2. start embedding
        self.T = sum(s for s in scales)
        init_std = math.sqrt(1 / latent_dim / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_t, latent_dim))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. motion length embedding
        self.len_embed = nn.Embedding(self.opt.max_motion_length // self.opt.unit_length , latent_dim)
        nn.init.trunc_normal_(self.len_embed.weight.data, mean=0, std=init_std)

        # 4. absolute position embedding
        # pos_1LC = []
        # for i, s in enumerate(self.scales):
        #     pe = torch.empty(1, s, latent_dim)
        #     nn.init.trunc_normal_(pe, mean=0, std=init_std)
        #     pos_1LC.append(pe)
        # pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, T, embed_dim
        # assert tuple(pos_1LC.shape) == (1, self.T, latent_dim)
        # self.pos_embedding = nn.Parameter(pos_1LC)
        scale_freqs_cis=[]
        self.rope_norm = latent_dim // num_heads
        self.compute_cis = partial(compute_cis, dim=latent_dim//num_heads, theta=100, normalize=self.rope_norm)
        for i, pn in enumerate(self.scales):
            freqs_cis = self.compute_cis(end_x = pn)
            scale_freqs_cis.append(freqs_cis)
        self.register_buffer('freqs_cis', torch.cat(scale_freqs_cis, dim=0))#(L,latent_dim//head)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.scales), latent_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 5. backbone blocks
        self.blocks = nn.ModuleList([Block(self.latent_dim, num_heads, dropout, ff_size) for _ in range(num_layers)])

        # 6. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        lvl_1L: torch.Tensor = torch.cat([torch.full((s, ), i) for i, s in enumerate(self.scales)]).view(1, self.T)
        self.register_buffer('lvl_1L', lvl_1L)

        d: torch.Tensor = torch.cat([torch.full((s, ), i) for i, s in enumerate(self.scales)]).view(1, self.T, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 1., 0.).reshape(1, 1, self.T, self.T)
        self.register_buffer('attn_mask', attn_bias_for_masking.contiguous())

        # 7. classifier head
        # self.head_nm = AdaLNBeforeHead(self.latent_dim, self.clip_dim)
        nb_codes = [round(nb_code) for nb_code in np.linspace(nb_code_st, nb_code_ed, len(self.scales))]
        self.heads = nn.ModuleList([nn.Linear(self.latent_dim, nb_codes[i]) for i in range(len(self.scales))])

        # 8. init networks
        self.apply(self.__init_weights)

        # 9. loading clip
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = CLIPModelWrapper(self.clip_version)


    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def cond_mask(self, cond, force_mask=False):
        bs =  cond.shape[0]
        if force_mask:
            return torch.zeros(bs, device=cond.device)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob)
            return (1. - mask)
        else:
            return torch.ones(bs, device=cond.device)

    def forward(self, x_inputs, y, x_labels, m_lens, p_drop_factor=0.0):

        bs = len(y)
        device = next(self.parameters()).device

        Ts = m_lens // self.opt.unit_length

        ######### [Important! the latent tokens no longer need pad mask anymore!!!] ######
        # padding_mask = ~lengths_to_mask(Ts, self.scales[-1])
        # padding_mask_scales = []
        # for i, q in enumerate(self.scales):
        #     mask = padding_mask[:, :q]
        #     # These pad position is no need for loss optimization.
        #     x_labels[i][mask] = self.pad_id
        #     padding_mask_scales.append(mask)
        # padding_mask = torch.concatenate(padding_mask_scales, dim=1)

        '''
        Preparing Input
        '''
        len_embed = self.len_embed((m_lens // self.opt.unit_length)-1)
        with torch.no_grad():
            sent_vector, word_vector, word_mask = self.clip_model(y) # [b, t], [b, t, 512]
        cond_mask = self.cond_mask(word_vector)
        cond = self.input_emb(word_vector * cond_mask[:, None, None]) #(b, t, latent_dim)
        word_mask[cond_mask==0] = True
        cond += len_embed.unsqueeze(1)

        # the [s] token: (b, seqlen, d)
        sos = torch.zeros(bs, self.first_t, self.latent_dim).float().to(device)
        sos += self.pos_start.expand(bs, self.first_t, -1)
        # sos += self.input_emb(sent_vector * cond_mask[:, None]).unsqueeze(1)
        sos += len_embed.unsqueeze(1)
        if x_inputs is not None:
            x = self.input_process(x_inputs)
            xseq = torch.cat([sos, x], dim=1) #(b, seqlen+self.first_t, latent_dim)
        else:
            xseq = sos

        # When progressive training is enabled, the xseq is no longer the flattened full-scale tokens.
        # Therefore, we need to truncate the position encoding and attn mask.
        cur_T = xseq.size(1)
        xseq = xseq + self.lvl_embed(self.lvl_1L[:, :cur_T]) # + self.pos_embedding[:, :cur_T]
        attn_mask = self.attn_mask[:, :, :cur_T, :cur_T]
        freqs_cis = self.freqs_cis
        '''
        trans_forward
        '''
        for block in self.blocks:
            # xseq = block(xseq, cond, freqs_cis, attn_mask, padding_mask=~padding_mask)
            xseq = block(xseq, cond, freqs_cis, attn_mask)
        '''
        prediction
        '''
        s_prev = 0
        logits = []
        for s, head in zip(self.scales, self.heads):
            logits.append(head(xseq[:,s_prev:s_prev+s].float()).float().permute(0, 2, 1))
            s_prev += s
        '''
        loss computation
        '''
        ce_losses, pred_ids, acc_scales = [], [], []
        for q, X in enumerate(x_labels):
            ce_loss, pred_id, acc = cal_performance_naive(logits[q], X, ignore_index=self.pad_id)
            ce_losses.append(ce_loss)
            pred_ids.append(pred_id)
            acc_scales.append(acc)
        
        ce_loss = (sum(ce_losses[:-1]) / len(ce_losses[:-1]) + ce_losses[-1]) if len(ce_losses) > 1 else ce_losses[-1]
        acc_total = sum(acc_scales) / len(acc_scales)

        return ce_loss, pred_id, acc_total, acc_scales
    

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 cond_scale: int,
                 vq_model,
                 top_k = 0.8,
                 top_p = 0.95,
                 temperature = 1.0,
                 more_smooth = False,
                 return_ids = False,
                 return_last = False
                 ):

        device = next(self.parameters()).device
        bs = len(m_lens)

        '''
        Preparing input
        '''
        sent_vector, word_vector, word_mask = self.clip_model(conds) # [b, t], [b, t, 512]
        len_embed = self.len_embed((m_lens // self.opt.unit_length)-1).repeat(2, 1)
        # classifier-free guidance
        cond_mask = self.cond_mask(word_vector, force_mask=True)
        word_vector = torch.cat((word_vector, word_vector * cond_mask[:, None, None]), dim=0)
        word_mask = torch.cat([word_mask, torch.ones_like(word_mask)], dim=0)
        cond = self.input_emb(word_vector)
        cond += len_embed.unsqueeze(1)

        Ts = m_lens // self.opt.unit_length
        # padding_mask = ~lengths_to_mask(Ts, self.scales[-1])
        # padding_mask = torch.concatenate([padding_mask[:, :pn] for pn in self.scales], dim=1)

        sos = torch.zeros(bs*2, self.first_t, self.latent_dim).float().to(device)
        sos += self.pos_start.expand(bs * 2, self.first_t, -1)
        # sent_vector = torch.cat((sent_vector, sent_vector * cond_mask[:, None]), dim=0)
        # sos += self.input_emb(sent_vector).unsqueeze(1)
        sos += len_embed.unsqueeze(1)

        pos_embd = self.lvl_embed(self.lvl_1L) # + self.pos_embedding

        next_token_map = sos + pos_embd[:, :self.first_t]
        
        z_hat = sos.new_zeros(bs, self.code_dim, self.scales[-1])

        for b in self.blocks: b.kv_caching(True)

        cur_T = 0
        all_pred_ids = []
        for q, pn in enumerate(self.scales):
            ratio = q / self.num_stages_minus1
            xseq = next_token_map
            cur_T += pn
            freqs_cis_cur = self.freqs_cis[cur_T-pn: cur_T,:]
            # (b, num_token, seqlen)
            for block in self.blocks:
                # xseq = block(xseq, cond, freqs_cis_cur, padding_mask=~padding_mask[:, :cur_T].repeat(2, 1))
                xseq = block(xseq, cond, freqs_cis_cur)
            
            # logits = self.heads[s](self.head_nm(xseq.float(), sent_vector_bf).float()).float().permute(0, 2,1)
            logits = self.heads[q](xseq.float()).float().permute(0, 2,1)
            # classifier-free guidance ratio
            t = cond_scale * ratio
            logits = (1+t) * logits[:bs] - t * logits[bs:]

            logits = logits.permute(0, 2, 1).contiguous()  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            pred_ids = sample_with_top_k_top_p_(logits, rng=None, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1)[:, :, 0]
            all_pred_ids.append(pred_ids)

            '''
            Preparing next token input
            '''
            if not more_smooth:
                # z_BCs = vq_model.quantizer.layers[q].dequantize(pred_ids).contiguous()
                z_BCs =  F.embedding(pred_ids, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                z_BCs = gumbel_softmax_with_rng(logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=None) @ vq_model.quantizer.layers[q].codebook.unsqueeze(0)
                z_BCs = z_BCs.permute(0, 2, 1).contiguous()
            z_hat, next_token_map = vq_model.quantizer.get_next_autoregressive_input(q, z_hat, z_BCs)
            if q < self.num_stages_minus1:   # prepare for next stage
                # NOTE fix bug: don't use view!!!
                # next_token_map = next_token_map.view(bs, self.code_dim, -1).transpose(1, 2)
                next_token_map = self.input_process(next_token_map) + pos_embd[:, cur_T:cur_T + self.scales[q+1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.kv_caching(False)
        if return_ids:
            return z_hat, all_pred_ids
        if return_last:
            return z_hat, xseq[:bs], z_BCs
        return z_hat


    @torch.no_grad()
    @eval_decorator
    def sar_conditional_entropy(self,
                                conds,
                                m_lens,
                                target_ids_list,   
                                vq_model,
                                cond_scale: float = 0.0
                                ):
        """
        计算每个尺度的困惑度（基于预测token分布的熵）。
        
        返回每个尺度的困惑度列表。

        target_ids_list[q] 的形状必须与该尺度的 logits 的 (bs, seqlen) 对齐。
        如果你只有拼接后的 token 序列，需要按 self.scales 切分成每尺度的列表。
        """
        device = next(self.parameters()).device
        bs = len(m_lens)
        assert isinstance(target_ids_list, (list, tuple)), "target_ids_list must be a list of per-scale target ids"
        assert len(target_ids_list) == len(self.scales), "target_ids_list length must equal number of scales"

        # 1) 文本与条件准备（与 generate 保持一致）
        sent_vector, word_vector, word_mask = self.clip_model(conds)  # [b, t], [b, t, 512]
        len_embed = self.len_embed((m_lens // self.opt.unit_length) - 1).repeat(2, 1)

        # classifier-free guidance 的无条件分支
        cond_mask = self.cond_mask(word_vector, force_mask=True)
        word_vector = torch.cat((word_vector, word_vector * cond_mask[:, None, None]), dim=0)
        word_mask = torch.cat([word_mask, torch.ones_like(word_mask)], dim=0)

        cond = self.input_emb(word_vector)
        cond += len_embed.unsqueeze(1)

        # 2) 起始 token map（与 generate 保持一致）
        sos = torch.zeros(bs * 2, self.first_t, self.latent_dim, dtype=torch.float32, device=device)
        sos += self.pos_start.expand(bs * 2, self.first_t, -1)
        sos += len_embed.unsqueeze(1)

        pos_embd = self.lvl_embed(self.lvl_1L)  # 位置/层级嵌入
        next_token_map = sos + pos_embd[:, :self.first_t]

        # 3) 遍历尺度，计算 logits 与条件交叉熵
        ppl_list = []  # 每尺度一个张量，形状 [B] 或 [B, T_n] 取决于 reduce
        z_hat = sos.new_zeros(bs, self.code_dim, self.scales[-1])
        for b in self.blocks:
            b.kv_caching(True)

        cur_T = 0
        for q, pn in enumerate(self.scales):
            ratio = q / self.num_stages_minus1 if self.num_stages_minus1 > 0 else 0.0
            xseq = next_token_map
            cur_T += pn
            freqs_cis_cur = self.freqs_cis[cur_T - pn: cur_T, :]

            # Transformer blocks 前向
            for block in self.blocks:
                xseq = block(xseq, cond, freqs_cis_cur)

            # 头部得到 logits（与 generate 同步）
            logits = self.heads[q](xseq.float()).float().permute(0, 2, 1)  # (bs*2, ntoken, seqlen)

            # CFG 融合（保持与生成一致）
            t = float(cond_scale) * ratio
            logits_mixed = (1.0 + t) * logits[:bs] - t * logits[bs:]      # (bs, ntoken, seqlen)
            logits_mixed = logits_mixed.permute(0, 2, 1).contiguous()     # (bs, seqlen, ntoken)

            # 目标 token ids（vq 之后的离散 id），用来计算条件交叉熵
            targets = target_ids_list[q].to(device)
            assert targets.shape[:2] == logits_mixed.shape[:2], \
                f"Targets shape {targets.shape} and logits shape {logits_mixed.shape} must match on (bs, seqlen)"

            # 计算困惑度：基于预测token分布的熵
            # logits_mixed: (bs, seqlen, vocab_size)
            vocab_size = logits_mixed.shape[-1]
            code_idx = logits_mixed.argmax(dim=-1).view(-1)
            
            code_onehot = torch.zeros(vocab_size, code_idx.shape[0], device=code_idx.device)  # V, N * L
            code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

            code_count = code_onehot.sum(dim=-1)  # V
            prob = code_count / torch.sum(code_count)  
            perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            ppl_list.append(perplexity)

            # 用真实 targets 作为下一级输入，确保“条件在已知上级尺度”的评估
            z_BCs = F.embedding(targets, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
            z_hat, next_token_map = vq_model.quantizer.get_next_autoregressive_input(q, z_hat, z_BCs)

            # 准备下一尺度的 token map（与 generate 一致）
            if q < self.num_stages_minus1:
                next_token_map = self.input_process(next_token_map) + pos_embd[:, cur_T: cur_T + self.scales[q + 1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # 由于 CFG，需要双倍 batch

        for b in self.blocks:
            b.kv_caching(False)

        return ppl_list  # list of length N_scales; each [B] if reduce='mean'

    # @torch.no_grad()
    # @eval_decorator
    # def generate(self,
    #              conds,
    #              m_lens,
    #              cond_scale: int,
    #              vq_model,
    #              top_k = 0.8,
    #              top_p = 0.95,
    #              temperature = 1.0,
    #              more_smooth = False,
    #              return_ids = False,
    #              return_last = False,
    #              tfg = 0.0,
    #              scale_ids = [],
    #              ):

    #     device = next(self.parameters()).device
    #     bs = len(m_lens)

    #     '''
    #     Preparing input
    #     '''
    #     sent_vector, word_vector, word_mask = self.clip_model(conds) # [b, t], [b, t, 512]
    #     len_embed = self.len_embed((m_lens // self.opt.unit_length)-1).repeat(2, 1)
    #     # classifier-free guidance
    #     cond_mask = self.cond_mask(word_vector, force_mask=True)
    #     word_vector = torch.cat((word_vector, word_vector * cond_mask[:, None, None]), dim=0)
    #     word_mask = torch.cat([word_mask, torch.ones_like(word_mask)], dim=0)
    #     cond = self.input_emb(word_vector)
    #     cond += len_embed.unsqueeze(1)

    #     Ts = m_lens // self.opt.unit_length
    #     # padding_mask = ~lengths_to_mask(Ts, self.scales[-1])
    #     # padding_mask = torch.concatenate([padding_mask[:, :pn] for pn in self.scales], dim=1)

    #     sos = torch.zeros(bs*2, self.first_t, self.latent_dim).float().to(device)
    #     sos += self.pos_start.expand(bs * 2, self.first_t, -1)
    #     # sent_vector = torch.cat((sent_vector, sent_vector * cond_mask[:, None]), dim=0)
    #     # sos += self.input_emb(sent_vector).unsqueeze(1)
    #     sos += len_embed.unsqueeze(1)

    #     pos_embd = self.lvl_embed(self.lvl_1L) # + self.pos_embedding

    #     next_token_map = sos + pos_embd[:, :self.first_t]
        
    #     z_hat = sos.new_zeros(bs, self.code_dim, self.scales[-1])

    #     for b in self.blocks: b.kv_caching(True)

    #     cur_T = 0
    #     all_pred_ids = []
        
    #     perplexitys = []
    #     for q, pn in enumerate(self.scales):
    #         ratio = q / self.num_stages_minus1
    #         xseq = next_token_map
    #         cur_T += pn
    #         freqs_cis_cur = self.freqs_cis[cur_T-pn: cur_T,:]
    #         # (b, num_token, seqlen)
    #         for block in self.blocks:
    #             # xseq = block(xseq, cond, freqs_cis_cur, padding_mask=~padding_mask[:, :cur_T].repeat(2, 1))
    #             xseq = block(xseq, cond, freqs_cis_cur)
            
    #         # logits = self.heads[s](self.head_nm(xseq.float(), sent_vector_bf).float()).float().permute(0, 2,1)
    #         logits = self.heads[q](xseq.float()).float().permute(0, 2,1)
    #         # classifier-free guidance ratio
    #         t = cond_scale * ratio
    #         logits = (1+t) * logits[:bs] - t * logits[bs:]

    #         gt_ids = scale_ids[q]  # (b, seqlen)
    #         logp = F.log_softmax(logits, dim=1)  # (b, vocab_size, seqlen)

    #         # 正确的维度处理
    #         batch_size, vocab_size, seq_len = logp.shape
    #         # gt_ids: (b, seqlen) -> 需要转换为 (b, seqlen, 1) 来gather
    #         gt_ids_expanded = gt_ids.unsqueeze(2)  # (b, seqlen, 1)

    #         # 转换logp为 (b, seqlen, vocab_size) 来匹配gather的维度要求
    #         logp = logp.permute(0, 2, 1)  # (b, seqlen, vocab_size)

    #         # 计算每个位置的negative log likelihood
    #         nll = -logp.gather(2, gt_ids_expanded).squeeze(2)  # (b, seqlen)

    #         # 计算平均NLL和困惑度
    #         nll_mean = torch.mean(nll, dim=1)  # (b,) - 每个序列的平均NLL
    #         perplexity = torch.exp(nll_mean)  # (b,)
    #         perplexitys.append(perplexity.mean().item())

    #         logits = logits.permute(0, 2, 1).contiguous()  # (b, seqlen, ntoken)
    #         # print(logits.shape, self.opt.num_tokens)
    #         pred_ids = sample_with_top_k_top_p_(logits, rng=None, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1)[:, :, 0]
    #         all_pred_ids.append(pred_ids)

    #         '''
    #         Preparing next token input
    #         '''
    #         if not more_smooth:
    #             # z_BCs = vq_model.quantizer.layers[q].dequantize(pred_ids).contiguous()
    #             z_BCs =  F.embedding(pred_ids, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
    #         else:  # not used when evaluating FID/IS/Precision/Recall
    #             gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
    #             z_BCs = gumbel_softmax_with_rng(logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=None) @ vq_model.quantizer.layers[q].codebook.unsqueeze(0)
    #             z_BCs = z_BCs.permute(0, 2, 1).contiguous()
    #         z_hat, next_token_map = vq_model.quantizer.get_next_autoregressive_input(q, z_hat, z_BCs)
    #         if q < self.num_stages_minus1:   # prepare for next stage
    #             # NOTE fix bug: don't use view!!!
    #             # next_token_map = next_token_map.view(bs, self.code_dim, -1).transpose(1, 2)
    #             next_token_map = self.input_process(next_token_map) + pos_embd[:, cur_T:cur_T + self.scales[q+1]]
    #             next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

    #     for b in self.blocks: b.kv_caching(False)
    #     if return_ids:
    #         return z_hat, all_pred_ids
    #     if return_last:
    #         return z_hat, xseq[:bs], z_BCs
    #     return z_hat, sum(perplexitys)/len(perplexitys)
    

    @torch.no_grad()
    @eval_decorator
    def edit(self,
                 conds,
                 gt_tokens,
                 edit_mask,
                 m_lens,
                 cond_scale: int,
                 vq_model,
                 top_k = 0.8,
                 top_p = 0.95,
                 temperature = 1.0,
                 more_smooth = False,
                 force_fill_q = -1,
                 condition_free = False,
                 ):

        device = next(self.parameters()).device
        bs = len(m_lens)

        '''
        Preparing input
        '''
        sent_vector, word_vector, word_mask = self.clip_model(conds) # [b, t], [b, t, 512]
        len_embed = self.len_embed((m_lens // self.opt.unit_length)-1).repeat(2, 1)
        # classifier-free guidance
        cond_mask = self.cond_mask(word_vector, force_mask=True)
        word_vector = torch.cat((word_vector, word_vector * cond_mask[:, None, None]), dim=0)
        word_mask = torch.cat([word_mask, torch.ones_like(word_mask)], dim=0)
        cond = self.input_emb(word_vector)
        cond += len_embed.unsqueeze(1)

        sos = torch.zeros(bs*2, self.first_t, self.latent_dim).float().to(device)
        sos += self.pos_start.expand(bs * 2, self.first_t, -1)
        # sent_vector = torch.cat((sent_vector, sent_vector * cond_mask[:, None]), dim=0)
        # sos += self.input_emb(sent_vector).unsqueeze(1)
        sos += len_embed.unsqueeze(1)

        pos_embd = self.lvl_embed(self.lvl_1L) # + self.pos_embedding

        next_token_map = sos + pos_embd[:, :self.first_t]
        
        z_hat = sos.new_zeros(bs, self.code_dim, self.scales[-1])

        for b in self.blocks: b.kv_caching(True)

        cur_T = 0
        for q, pn in enumerate(self.scales):
            ratio = q / self.num_stages_minus1
            xseq = next_token_map
            cur_T += pn
            freqs_cis_cur = self.freqs_cis[cur_T-pn: cur_T,:]
            # (b, num_token, seqlen)
            for block in self.blocks:
                xseq = block(xseq, cond, freqs_cis_cur)
            
            # logits = self.heads[s](self.head_nm(xseq.float(), sent_vector_bf).float()).float().permute(0, 2,1)
            logits = self.heads[q](xseq.float()).float().permute(0, 2,1)
            # classifier-free guidance ratio
            t = cond_scale * ratio
            if not condition_free:
                logits = (1+t) * logits[:bs] - t * logits[bs:]
            else:
                logits = t * logits[bs:]

            logits = logits.permute(0, 2, 1).contiguous()  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            pred_ids = sample_with_top_k_top_p_(logits, rng=None, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1)[:, :, 0]

            '''
            Preparing next token input
            '''
            if not more_smooth:
                z_BCs =  F.embedding(pred_ids, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                z_BCs = gumbel_softmax_with_rng(logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=None) @ vq_model.quantizer.layers[q].codebook.unsqueeze(0)
                z_BCs = z_BCs.permute(0, 2, 1).contiguous()
            
            '''
            **key difference**
            '''
            if edit_mask is not None:
                gt_z_BCs =  F.embedding(gt_tokens[q], vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
                scale_edit_mask = F.interpolate(edit_mask.view(bs, 1, -1), size=pn, mode='linear') > 0.5
                if q<= force_fill_q:
                    # force fill
                    scale_edit_mask = torch.zeros_like(scale_edit_mask).bool()
                z_BCs = torch.where(scale_edit_mask, z_BCs, gt_z_BCs)
            else:
                # pad_id = -1
                pred_ids = torch.where(gt_tokens[q]==-1, pred_ids, gt_tokens[q])
                z_BCs =  F.embedding(pred_ids, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
            
            z_hat, next_token_map = vq_model.quantizer.get_next_autoregressive_input(q, z_hat, z_BCs)
            if q < self.num_stages_minus1:   # prepare for next stage
                # NOTE fix bug: don't use view!!!
                # next_token_map = next_token_map.view(bs, self.code_dim, -1).transpose(1, 2)
                next_token_map = self.input_process(next_token_map) + pos_embd[:, cur_T:cur_T + self.scales[q+1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.kv_caching(False)
        return z_hat
    

    @torch.no_grad()
    @eval_decorator
    def long_range(self,
                 conds,
                 tokens,
                 m_lens,
                 cond_scale: int,
                 vq_model,
                 top_k = 0.8,
                 top_p = 0.95,
                 temperature = 1.0,
                 more_smooth = False,
                 transition_n_token = 3, 
                 ):
        
        device = next(self.parameters()).device
        bs = len(m_lens)
        token_lens = m_lens // 4

        if tokens is None:
            z_hat, tokens = self.generate(conds, m_lens, cond_scale, vq_model, \
                                  top_k, top_p, temperature, more_smooth, return_ids=True)
        
        last_scale = tokens[-1].shape[1]
        motion_n_token = (last_scale - transition_n_token) // 2
        pad_id = -1

        z_hat_mids = torch.zeros((bs-1, z_hat.shape[1],last_scale), device=z_hat.device)
        total_frame = 0

        for i in range(bs-1):
            left_mask = torch.zeros(last_scale)
            right_mask = torch.zeros(last_scale)
            right_region_mask = torch.zeros(last_scale)
            edit_mask = torch.zeros(last_scale).cuda()
            left_part_end = token_lens[i] 
            left_part_start = max(token_lens[i] - motion_n_token, 0)
            left_part_duration = left_part_end - left_part_start
            left_mask[left_part_start:left_part_end] = 1.0

            right_part_duration = last_scale - (left_part_duration + transition_n_token) 
            right_mask[:right_part_duration] = 1.0
            right_region_mask[left_part_duration+transition_n_token:] = 1.0
            edit_mask[left_part_duration:left_part_duration+transition_n_token] = 1.0

            new_tokens = []
            for scale_tokens in tokens:
                pn = scale_tokens.shape[1]
                left_part_tokens, right_part_tokens = scale_tokens[i:i+1], scale_tokens[i+1:i+2]

                new_scale_token = torch.full((1, pn), pad_id).to(scale_tokens.device)
                sclae_left_mask = F.interpolate(left_mask.view(1, 1, -1), size=pn, mode='linear').squeeze(dim=1) > 0.5
                sclae_right_mask = F.interpolate(right_mask.view(1, 1, -1), size=pn, mode='linear').squeeze(dim=1) > 0.5
                sclae_right_region_mask = F.interpolate(right_region_mask.view(1, 1, -1), size=pn, mode='linear').squeeze(dim=1) > 0.5
                
                new_scale_token[:, :sclae_left_mask.sum()] = left_part_tokens[sclae_left_mask]
                new_scale_token[sclae_right_region_mask] = right_part_tokens[sclae_right_mask]

                new_tokens.append(new_scale_token)
            
            z_hat_mid = self.edit(conds[i:i+1], new_tokens,  None, torch.tensor([196,]).cuda(), cond_scale, vq_model, \
                                  top_k, top_p, temperature, more_smooth, condition_free=True)
            z_hat_mids[i] = z_hat_mid

        new_z_hat = []
        for i in range(bs):
            new_z_hat.append(z_hat[i])
            if i < bs - 1:
                new_z_hat.append(z_hat_mids[i])
        new_z_hat = torch.stack(new_z_hat, dim=0)

        # 49 49 49
        # [0, 26] [0, 49] [23, 26], [0, 49], [24, 49]
        frame_infos = [
            (0, 26),(0,49), (23, 26), (0, 49), (23,49)
        ]

        return new_z_hat



        

        


    @torch.no_grad()
    @eval_decorator
    def generate_at_scale(self, q,
                 conds,
                 m_lens,
                 cond_scale: int,
                 vq_model,
                 top_k = 0.8,
                 top_p = 0.95,
                 temperature = 1.0):

        device = next(self.parameters()).device
        bs = len(m_lens)

        assert q < len(self.scales)

        '''
        Preparing input
        '''
        sent_vector, word_vector, word_mask = self.clip_model(conds) # [b, t], [b, t, 512]
        len_embed = self.len_embed((m_lens // self.opt.unit_length)-1).repeat(2, 1)
        # classifier-free guidance
        cond_mask = self.cond_mask(word_vector, force_mask=True)
        word_vector = torch.cat((word_vector, word_vector * cond_mask[:, None, None]), dim=0)
        cond = self.input_emb(word_vector)
        cond += len_embed.unsqueeze(1)

        sos = torch.zeros(bs*2, self.first_t, self.latent_dim).float().to(device)
        sos += self.pos_start.expand(bs * 2, self.first_t, -1)
        # sent_vector = torch.cat((sent_vector, sent_vector * cond_mask[:, None]), dim=0)
        # sos += self.input_emb(sent_vector).unsqueeze(1)
        sos += len_embed.unsqueeze(1)

        pos_embd = self.lvl_embed(self.lvl_1L) # + self.pos_embedding

        next_token_map = sos + pos_embd[:, :self.first_t]

        scales = self.scales[:q+1]
        num_stages_minus1 = len(scales) - 1
        
        z_hat = sos.new_zeros(bs, self.code_dim, scales[-1])

        for b in self.blocks: b.kv_caching(True)
        
        cur_T = 0
        for q, pn in enumerate(scales):
            ratio = q / max(num_stages_minus1, 1e-4)
            xseq = next_token_map
            cur_T += pn
            freqs_cis_cur = self.freqs_cis[cur_T-pn: cur_T,:]
            # (b, num_token, seqlen)
            for block in self.blocks:
                xseq = block(xseq, cond, freqs_cis_cur)
            
            # logits = self.heads[s](self.head_nm(xseq.float(), sent_vector_bf).float()).float().permute(0, 2,1)
            logits = self.heads[q](xseq.float()).float().permute(0, 2,1)
            # classifier-free guidance ratio
            t = cond_scale * ratio
            logits = (1+t) * logits[:bs] - t * logits[bs:]

            logits = logits.permute(0, 2, 1).contiguous()  # (b, seqlen, ntoken)
            # _, pred_ids = torch.topk(F.softmax(logits, dim=-1), k=1, dim=-1)
            # pred_ids = pred_ids[:, :, 0]
            # print(logits.shape, self.opt.num_tokens)
            pred_ids = sample_with_top_k_top_p_(logits, rng=None, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1)[:, :, 0]

            '''
            Preparing next token input
            '''
            z_BCs =  F.embedding(pred_ids, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
            z_hat, next_token_map = vq_model.quantizer.get_next_autoregressive_input(q, z_hat, z_BCs, scales)
            if q < num_stages_minus1:   # prepare for next stage
                # NOTE fix bug: don't use view!!!
                # next_token_map = next_token_map.view(bs, self.code_dim, -1).transpose(1, 2)
                next_token_map = self.input_process(next_token_map) + pos_embd[:, cur_T:cur_T + scales[q+1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.kv_caching(False)
        return z_hat
    
    @torch.no_grad()
    @eval_decorator
    def generate_at_scale2(self,
                 conds,
                 m_lens,
                 cond_scale: int,
                 vq_model,
                 top_k = 0.8,
                 top_p = 0.95,
                 temperature = 1.0
                 ):

        device = next(self.parameters()).device
        bs = len(m_lens)

        '''
        Preparing input
        '''
        sent_vector, word_vector, word_mask = self.clip_model(conds) # [b, t], [b, t, 512]
        len_embed = self.len_embed((m_lens // self.opt.unit_length)-1).repeat(2, 1)
        # classifier-free guidance
        cond_mask = self.cond_mask(word_vector, force_mask=True)
        word_vector = torch.cat((word_vector, word_vector * cond_mask[:, None, None]), dim=0)
        word_mask = torch.cat([word_mask, torch.ones_like(word_mask)], dim=0)
        cond = self.input_emb(word_vector)
        cond += len_embed.unsqueeze(1)

        sos = torch.zeros(bs*2, self.first_t, self.latent_dim).float().to(device)
        sos += self.pos_start.expand(bs * 2, self.first_t, -1)
        # sent_vector = torch.cat((sent_vector, sent_vector * cond_mask[:, None]), dim=0)
        # sos += self.input_emb(sent_vector).unsqueeze(1)
        sos += len_embed.unsqueeze(1)

        pos_embd = self.lvl_embed(self.lvl_1L) # + self.pos_embedding

        next_token_map = sos + pos_embd[:, :self.first_t]
        
        z_hat = sos.new_zeros(bs, self.code_dim, self.scales[-1])
        inter_z_hats = []

        for b in self.blocks: b.kv_caching(True)

        cur_T = 0
        for q, pn in enumerate(self.scales):
            ratio = q / self.num_stages_minus1
            xseq = next_token_map
            cur_T += pn
            freqs_cis_cur = self.freqs_cis[cur_T-pn: cur_T,:]
            # (b, num_token, seqlen)
            for block in self.blocks:
                xseq = block(xseq, cond, freqs_cis_cur)
            
            # logits = self.heads[s](self.head_nm(xseq.float(), sent_vector_bf).float()).float().permute(0, 2,1)
            logits = self.heads[q](xseq.float()).float().permute(0, 2,1)
            # classifier-free guidance ratio
            t = cond_scale * ratio
            logits = (1+t) * logits[:bs] - t * logits[bs:]

            logits = logits.permute(0, 2, 1).contiguous()  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            pred_ids = sample_with_top_k_top_p_(logits, rng=None, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1)[:, :, 0]

            '''
            Preparing next token input
            '''
            z_BCs =  F.embedding(pred_ids, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
            z_hat, next_token_map = vq_model.quantizer.get_next_autoregressive_input(q, z_hat, z_BCs)
            inter_z_hats.append(F.interpolate(z_hat, size=(self.scales[q]), mode='area'))
            if q < self.num_stages_minus1:   # prepare for next stage
                # NOTE fix bug: don't use view!!!
                # next_token_map = next_token_map.view(bs, self.code_dim, -1).transpose(1, 2)
                next_token_map = self.input_process(next_token_map) + pos_embd[:, cur_T:cur_T + self.scales[q+1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.kv_caching(False)
        return inter_z_hats
