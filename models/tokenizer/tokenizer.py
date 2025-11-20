from typing import List
import torch
import torch.nn as nn

from models.tokenizer.quantizers.vq import ScaleVQ, ResidualVQ, IdentityVQ
from models.tokenizer.encdec import Encoder, Decoder
from utils.tools import lengths_to_mask

from models.tokenizer.text_wrapper import TextEmbedding

class Tokenizer(nn.Module):
    def __init__(self, args, 
                num_latent_tokens, 
                input_width, 
                latent_dim = 256,
                depth = 9, 
                dropout = 0.1,
                ff_size = 1024, 
                activation = 'gelu',
                patch_size = 1,
                norm_first = False,
                norm = 'RMS',
                qk_norm = False,
                pos_embed = 'rope1d',
                rope_base = 100,
                enc_moiton_text_embed = "ctx_ctx",
                dec_latent_text_embed = "crs_crs",
                cond_drop_prob = 0.1,
                mae_motion_drop = 0.4,
                mae_motion_drop_max = 0.6,
                tfg = 1.0,
                mfg = 1.0,
                code_dim = 512, 
                text_model = 'clip',
                text_max_len = 120,
                quant_type = 'mosa',
                max_motion_length = 196
        ):

        super().__init__()
        
        self.cond_drop_prob = cond_drop_prob
        self.patch_size = patch_size
        self.unit_length = args.unit_length
        self.joints_num = args.joints_num

        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.tfg = tfg
        self.mfg = mfg
        self.enc_text_embed = enc_moiton_text_embed.split('_')[-1]
        self.dec_text_embed = dec_latent_text_embed.split('_')[-1]

        scale = self.latent_dim ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.latent_dim))
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.latent_dim))

        self.text_model = TextEmbedding(args, text_model, text_max_len)
        self.text_len = self.text_model.max_length
        
        self.text_embed = nn.Linear(self.text_model.embed_dim, latent_dim)
        self.encoder = Encoder(num_latent_tokens=num_latent_tokens, text_length=self.text_len, enc_moiton_text_embed=enc_moiton_text_embed, ch=latent_dim, patch_size=self.patch_size, num_layers=depth, dropout=dropout, \
                              in_channels=input_width, z_channels=code_dim, ff_size=ff_size, activation=activation, norm_first=norm_first, norm=norm, qk_norm=qk_norm, pos_embed=pos_embed, rope_base=rope_base, \
                              token_drop=mae_motion_drop, token_drop_max=mae_motion_drop_max, max_motion_length=max_motion_length)
        self.decoder = Decoder(num_latent_tokens=num_latent_tokens, text_length=self.text_len, dec_latent_text_embed=dec_latent_text_embed, ch=latent_dim, patch_size=self.patch_size, num_layers=depth, dropout=dropout, \
                              in_channels=input_width, z_channels=code_dim, ff_size=ff_size, activation=activation, norm_first=norm_first, norm=norm, qk_norm=qk_norm, pos_embed=pos_embed, rope_base=rope_base, max_motion_length=max_motion_length) 
        
        if quant_type == 'mosa':
            # Multi-scale Quantization
            sqvae_config = {
                "scales": args.scales,
                "nb_code_st": args.nb_code_st,
                "nb_code_ed": args.nb_code_ed,
                "code_dim": code_dim,
                "args": args,
                "shared_codebook": args.shared_codebook,
                "phi_k": args.phi_k,
                "phi_depth": args.phi_depth,
            }

            assert self.num_latent_tokens == sqvae_config['scales'][-1], "num_latent_tokens must be equal to the last scale in scales"

            self.quantizer = ScaleVQ(**sqvae_config)
            print(f"Using Multi-scale Quantization with scales: {args.scales}, nb_code_st: {args.nb_code_st}, nb_code_ed: {args.nb_code_ed}, code_dim: {code_dim}, quant_layers: {len(args.scales)}")
        elif quant_type == 'mar':
            self.quantizer = IdentityVQ()
        else:
            assert args.nb_code_st == args.nb_code_ed
            if quant_type == 't2m-gpt':
                assert args.quant_layers == 1, "t2m-gpt only supports one quantization layer"
            vqvae_config = {
                "num_quantizers": args.quant_layers,
                "nb_code": args.nb_code_st,
                "code_dim": code_dim,
                "shared_codebook": args.shared_codebook,
                "quantize_dropout_prob": args.quantize_dropout_prob,
                'quantize_dropout_cutoff_index': 0,
                "sample_codebook_temp": args.sample_codebook_temp,
                "phi_k": args.phi_k,
                "phi_depth": args.phi_depth,
                "args": args,
            }
            # if quant_type == 't2m-gpt':
            #     self.quantizer = QuantizeEMAReset(args.nb_code_st, code_dim, args)
            # else:
            self.quantizer = ResidualVQ(**vqvae_config)
            print(f"Using Residual VQ with nb_code: {args.nb_code_st}, code_dim: {code_dim}, quant_layers: {args.quant_layers}")
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights.
        :param:
            module -> torch.nn.Module: module to initialize
        """
        if hasattr(self, "text_model") and any(module is m for m in self.text_model.modules()):
            return
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(prefix + 'text_model.')}
        return filtered_state_dict

    def preprocess(self, x, m_lens):
        x = x.float().contiguous()
        x = x[:, : max(m_lens)]
        mask = lengths_to_mask(m_lens, x.shape[1]).to(x.device)
        return x, mask

    def cond_mask(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros(bs, device=cond.device)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob)
            return (1. - mask)
        else:
            return torch.ones(bs, device=cond.device)

    def forward(self, m, m_lens, texts):
        m_lens = m_lens.to(m.device)

        text_feat, text_mask = self.text_model(texts)
        # cond mask for text
        text_cfg_mask = self.cond_mask(text_feat)
        motion_cfg_mask = self.cond_mask(m)
        # exclude overlaped indices
        motion_cfg_mask[text_cfg_mask + motion_cfg_mask == 0] = 1.0

        text_feat = self.text_embed(text_feat)
        text_feat = torch.where(text_cfg_mask.bool()[:, None, None], text_feat, self.mask_token)
        m, mask = self.preprocess(m, m_lens)
        text_enc_mask = text_mask & text_cfg_mask.bool()[:, None] if self.enc_text_embed == "ctx" else text_mask
        text_dec_mask = text_mask & text_cfg_mask.bool()[:, None] if self.dec_text_embed == "ctx" else text_mask

        # Encode
        z = self.forward_encoder(m, mask, text_feat, text_enc_mask, motion_cfg_mask)
        
        # Quantize
        z_hat, code_usage, commit_loss, perplexity, all_indices = self.quantizer(z)

        # Decode
        m_hat = self.forward_decoder(z_hat, mask, text_feat, text_dec_mask)

        return m_hat, code_usage, commit_loss, perplexity, all_indices
    

    def forward2quantize(self, m, m_lens, texts):
        m_lens = m_lens.to(m.device)

        text_feat, text_mask = self.text_model(texts)
        # cond mask for text
        text_cfg_mask = self.cond_mask(text_feat)
        motion_cfg_mask = self.cond_mask(m)
        # exclude overlaped indices
        motion_cfg_mask[text_cfg_mask + motion_cfg_mask == 0] = 1.0

        text_feat = self.text_embed(text_feat)
        text_feat = torch.where(text_cfg_mask.bool()[:, None, None], text_feat, self.mask_token)
        m, mask = self.preprocess(m, m_lens)
        text_enc_mask = text_mask & text_cfg_mask.bool()[:, None] if self.enc_text_embed == "ctx" else text_mask
        # Encode
        z = self.forward_encoder(m, mask, text_feat, text_enc_mask, motion_cfg_mask)
        
        # Quantize
        z_hat, code_usage, commit_loss, perplexity, all_indices = self.quantizer(z)

        return z_hat


    def forward_encoder(self, m, mask, text_feat, text_mask, motion_cfg_mask, mode='normal'):
        bs, seq_len = m.shape[:2]

        # with text_feat
        latent_tokens = self.latent_tokens.unsqueeze(0).expand(bs, -1, -1)

        latent_mask = mask.new_ones((bs, self.num_latent_tokens)).bool()

        z = self.encoder(m, latent_tokens, text_feat, mask, latent_mask, text_mask, motion_cfg_mask, mode)

        return z

    def forward_decoder(self, z_hat, mask, text_feat, text_mask):

        aug_mask = mask[:, :: self.patch_size]

        bs, seq_len = aug_mask.shape

        mask_tokens = self.mask_token.repeat(bs, seq_len, 1).to(z_hat.dtype)
        latent_mask = mask.new_ones((bs, self.num_latent_tokens)).bool()

        m_hat = self.decoder(mask_tokens, z_hat, text_feat, mask, latent_mask, text_mask)

        m_hat[~mask] = 0

        return m_hat
    
    def quantize(self, m, m_lens, texts, tfg=1.0, mfg=1.0):
        bs = len(m_lens)
        text_feat, text_mask = self.text_model(texts)
        # cond mask for text
        text_cfg_mask = self.cond_mask(text_feat)
        motion_cfg_mask = self.cond_mask(text_feat)
        # exclude overlaped indices
        motion_cfg_mask[text_cfg_mask + motion_cfg_mask == 0] = 1.0

        text_feat = self.text_embed(text_feat)
        text_feat = torch.where(text_cfg_mask.bool()[:, None, None], text_feat, self.mask_token)
        m, mask = self.preprocess(m, m_lens)
        text_enc_mask = text_mask & text_cfg_mask.bool()[:, None] if self.enc_text_embed == "ctx" else text_mask

        # Encode
        z = self.forward_encoder(m, mask, text_feat, text_enc_mask, motion_cfg_mask, mode='normal')

        code_idxs = self.quantizer.quantize(z)

        # bs = len(m)

        # text_feat, text_mask = self.text_model(texts)
        # # cond mask for text
        # text_mask1 = self.cond_mask(text_feat)
        # text_mask2 = self.cond_mask(text_feat, force_mask=True)
        # text_mask3 = self.cond_mask(text_feat)

        # text_cfg_mask = torch.cat([text_mask1, text_mask2, text_mask3], dim=0)

        # drop_mask = self.cond_mask(text_feat)
        # drop_mask2 = self.cond_mask(text_feat)
        # drop_mask3 = self.cond_mask(text_feat, force_mask=True)
        # drop_mask = torch.cat([drop_mask, drop_mask2, drop_mask3], dim=0)

        # text_feat = text_feat.repeat(3, 1, 1)
        # text_feat = self.text_embed(text_feat)
        # text_feat = torch.where(text_cfg_mask.bool()[:, None, None], text_feat, self.mask_token)
        # text_mask = text_mask.repeat(3, 1)
        # text_enc_mask = text_mask & text_cfg_mask.bool()[:, None] if self.enc_text_embed == "ctx" else text_mask

        # m = m.repeat(3, 1, 1)
        # m_lens = m_lens.repeat(3)
        # m, mask = self.preprocess(m, m_lens)

        # # Encode
        # z = self.forward_encoder(m, mask, text_feat, text_enc_mask, drop_mask, mode='masking')

        # # Quantize
        # z = (1+tfg+mfg) * z[:bs] - tfg * z[bs:2*bs] - mfg * z[2*bs:3*bs]
        # code_idxs = self.quantizer.quantize(z)

        return code_idxs
    
    def dequantize(self, z_hat, m_lens, texts, tfg=1.0):
        if z_hat.dtype in [torch.int64, torch.int, torch.int32]:
            z_hat = self.quantizer.get_codes_from_indices(z_hat)
            z_hat = z_hat.sum(dim=0).permute(0, 2, 1)

        bs = len(z_hat)

        text_feat, text_mask = self.text_model(texts)
        # ------------------
        # text_feat = self.text_embed(text_feat)
        # text_cfg_mask = self.cond_mask(text_feat, force_mask=True)
        # text_feat = torch.where(text_cfg_mask.bool()[:, None, None], text_feat, self.mask_token)
        # text_dec_mask = text_mask & text_cfg_mask.bool()[:, None] if self.dec_text_embed == "ctx" else text_mask
        # mask = lengths_to_mask(m_lens, max(m_lens)).to(z_hat.device)
        # m_hat = self.forward_decoder(z_hat, mask, text_feat, text_dec_mask)
        # m_hat, *_ = self.reconstruct(m_hat_tmp, m_lens, texts, tfg=tfg, mfg=0.2)
        # m_hat[:bs] = m_hat_tmp
        # mfg = 0.2
        # m_hat = (1+tfg+mfg) * m_hat[:bs] - tfg * m_hat[bs:2*bs] - mfg * m_hat[2*bs:3*bs]
        # ------------------
        # cond mask for text
        text_cfg_mask = self.cond_mask(text_feat)
        text_cfg_mask2 = self.cond_mask(text_feat, force_mask=True)
        text_cfg_mask = torch.cat([text_cfg_mask, text_cfg_mask2], dim=0)

        text_feat = text_feat.repeat(2, 1, 1)
        text_feat = self.text_embed(text_feat)
        text_feat = torch.where(text_cfg_mask.bool()[:, None, None], text_feat, self.mask_token)
        text_mask = text_mask.repeat(2, 1)
        text_dec_mask = text_mask & text_cfg_mask.bool()[:, None] if self.dec_text_embed == "ctx" else text_mask

        z_hat = z_hat.repeat(2, 1, 1)
        m_lens = m_lens.repeat(2)
        mask = lengths_to_mask(m_lens, max(m_lens)).to(z_hat.device)

        # Decode
        m_hat = self.forward_decoder(z_hat, mask, text_feat, text_dec_mask)
        m_hat = (1+tfg) * m_hat[:bs] - tfg * m_hat[bs:2*bs]
        
        # m_hat, *_ = self.reconstruction(m_hat, m_lens[:bs], texts)

        return m_hat


    def reconstruct(self, m, m_lens, texts, tfg=1.0, mfg=0.2):

        bs = len(m)

        text_feat, text_mask = self.text_model(texts)
        # cond mask for text
        text_mask1 = self.cond_mask(text_feat)
        text_mask2 = self.cond_mask(text_feat, force_mask=True)
        text_mask3 = self.cond_mask(text_feat)

        text_cfg_mask = torch.cat([text_mask1, text_mask2, text_mask3], dim=0)

        drop_mask = self.cond_mask(text_feat)
        drop_mask2 = self.cond_mask(text_feat)
        drop_mask3 = self.cond_mask(text_feat, force_mask=True)
        drop_mask = torch.cat([drop_mask, drop_mask2, drop_mask3], dim=0)

        text_feat = text_feat.repeat(3, 1, 1)
        text_feat = self.text_embed(text_feat)
        text_feat = torch.where(text_cfg_mask.bool()[:, None, None], text_feat, self.mask_token)
        text_mask = text_mask.repeat(3, 1)
        text_enc_mask = text_mask & text_cfg_mask.bool()[:, None] if self.enc_text_embed == "ctx" else text_mask
        text_dec_mask = text_mask & text_cfg_mask.bool()[:, None] if self.dec_text_embed == "ctx" else text_mask

        m = m.repeat(3, 1, 1)
        m_lens = m_lens.repeat(3)
        m, mask = self.preprocess(m, m_lens)

        # Encode
        z = self.forward_encoder(m, mask, text_feat, text_enc_mask, drop_mask, mode='masking')

        # Quantize
        # z = (1+tfg+mfg) * z[:bs] - tfg * z[bs:2*bs] - mfg * z[2*bs:3*bs]
        z_hat, code_usage, commit_loss, perplexity, all_indices = self.quantizer(z)

        # Decode
        m_hat = self.forward_decoder(z_hat, mask, text_feat, text_dec_mask)

        # w/ text + motion CFG
        # m_hat = (1-tfg-mfg) * m_hat[:bs] + tfg * m_hat[bs:2*bs] + mfg * m_hat[2*bs:3*bs]
        m_hat = (1+tfg+mfg) * m_hat[:bs] - tfg * m_hat[bs:2*bs] - mfg * m_hat[2*bs:3*bs]

        return m_hat, code_usage, commit_loss, perplexity, all_indices

# if __name__ == "__main__":
#     in_dim = 263
#     in_len = 196
#     ch = 128
#     num_res_blocks = 0
#     encoder = Encoder(ch = ch, ch_mult=(1, 1, 1), num_res_blocks=num_res_blocks, in_channels=in_dim, z_channels=ch, using_sa=False, using_mid_sa=False).cuda()
#     decoder = Decoder(ch = ch, ch_mult=(1, 1, 1), num_res_blocks=num_res_blocks, in_channels=in_dim, z_channels=ch, using_sa=False, using_mid_sa=False).cuda()
#     motion = torch.randn((32, in_dim, in_len)).cuda()
#     res = encoder(motion)
#     print(res.shape)
#     res = decoder(res)
#     print(res.shape)

#     pc_vq = sum(param.numel() for param in encoder.parameters() if param.requires_grad)
#     pc_vq += sum(param.numel() for param in decoder.parameters() if param.requires_grad)

#     print('Total parameters of all models: {}M'.format(pc_vq/1000_000))
