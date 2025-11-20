import math
import numpy as np
import torch
from torch import einsum, nn
import torch.nn.functional as F

from models.tokenizer.network import RMSNorm

class QuantizeInterpolatedEMAReset(nn.Module):
    def __init__(self, V, C, args):
        super().__init__()
        self.V = V
        self.C = C
        self.mu = args.mu
        self.using_znorm = args.using_znorm
        self.reset_codebook()
        
    def reset_codebook(self):
        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('code_sum', torch.zeros(self.V, self.C).cuda())
        self.register_buffer('codebook', torch.zeros(self.V, self.C).cuda())
        self.register_buffer('code_count', torch.ones(self.V, device=self.codebook.device))

    def _tile(self, x):
        V_x, C = x.shape
        if V_x < self.V:
            n_repeats = (self.V + V_x - 1) // V_x
            std = 0.01 / np.sqrt(C)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        # Preprocess
        x = self.merge_BT(x)
        out = self._tile(x)
        self.codebook = out[:self.V]
        self.code_sum = self.codebook.clone()
        self.initted = x.new_tensor([True])
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.V, code_idx.shape[0], device=code_idx.device)  # V, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # V
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        # Preprocess
        x = self.merge_BT(x)

        code_onehot = torch.zeros(self.V, x.shape[0], device=x.device)  # V, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # V, w
        code_count = code_onehot.sum(dim=-1)  # V

        out = self._tile(x)
        code_rand = out[:self.V]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, V
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # V

        usage = (self.code_count.view(self.V, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.V, self.C) / self.code_count.view(self.V, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def merge_BT(self, x):
        # BCT -> BTC -> [BT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, z, q):
        B, C, T = z.shape
        # downsampling
        rest_NC = F.interpolate(z, size=(q), mode='area').permute(0, 2, 1).reshape(-1, C) if (q != T) else z.permute(0, 2, 1).reshape(-1, C)
        if self.using_znorm:
            rest_NC = F.normalize(rest_NC, dim=-1)
            sim = rest_NC @ F.normalize(self.codebook.data.T, dim=0)
            idx_Bpn = torch.argmax(sim, dim=1).view(B, q)
        else:
            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.codebook.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(rest_NC, self.codebook.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_Bpn = torch.argmin(d_no_grad, dim=1).view(B, q)
        
        # We have observed that employing unsampled data for look-up and
        # appling EMA can lead to an increase in perplexity!
        z_tmp = z.permute(0, 2, 1).reshape(-1, C)
        if self.using_znorm:
            rest_NC = F.normalize(z_tmp, dim=-1)
            sim = rest_NC @ F.normalize(self.codebook.data.T, dim=0)
            idx_org = torch.argmax(sim, dim=1).view(-1)
        else:
            d_no_grad = torch.sum(z_tmp, dim=1, keepdim=True) + torch.sum(self.codebook.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_tmp, self.codebook.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_org = torch.argmin(d_no_grad, dim=1).view(-1)

        return idx_Bpn, idx_org
        
    def dequantize(self, code_idx, T, phi):
        z_hat = F.embedding(code_idx, self.codebook).permute(0, 2, 1).contiguous()

        # unsampling
        if code_idx.shape[1] != T:
            z_hat = F.interpolate(z_hat, size=(T), mode='linear').contiguous()

        z_hat = phi(z_hat)
        return z_hat
    
    def forward(self, z, q, phi, return_idx=False):
        B, width, T = z.shape

        code_idx, code_org = self.quantize(z, q)
        z_hat = self.dequantize(code_idx, T, phi)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(z, code_org)
        else : 
            perplexity = self.compute_perplexity(code_org)        

        if return_idx:
            return z_hat, code_idx, perplexity
        
        return z_hat, perplexity



class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim = 512,
        norm_type = None,
        attn_dim = None,
        n_head = 4,
    ):
        super().__init__()

        if attn_dim is None:
            attn_dim = hidden_dim

        self.to_q = nn.Linear(hidden_dim, attn_dim)
        self.to_k = nn.Linear(hidden_dim, attn_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)

        self.pool_proj = nn.Linear(hidden_dim, n_head)

        if norm_type is None:
            self.norm_q = None
            self.norm_k = None
        elif norm_type == 'layer_norm':
            self.norm_q = nn.LayerNorm(attn_dim // n_head)
            self.norm_k = nn.LayerNorm(attn_dim // n_head)
        elif norm_type == 'rms_norm':
            self.norm_q = RMSNorm(attn_dim // n_head, eps=1e-5)
            self.norm_k = RMSNorm(attn_dim // n_head, eps=1e-5)
        else:
            raise NotImplementedError

        self.n_head = n_head

    def forward(
        self,
        hidden_states,
        codebook_hidden_states,
        only_index = False,
    ):
        B, C, T = hidden_states.shape
        N, _ = codebook_hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 1)

        query = self.to_q(hidden_states).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b, 49, 512) -> (b, 49, 32)
        key = self.to_k(codebook_hidden_states).view(N, self.n_head, C // self.n_head).transpose(0, 1) # (1024 512) -> (1024, 32)
        value = self.to_v(codebook_hidden_states) # (1024 512) -> (1024, 32)

        c = self.pool_proj(hidden_states) # (B, T, nh)

        scale_factor = 1 / math.sqrt(query.size(-1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        logits = einsum('b h t d, h n d -> b h n t', query, key) * scale_factor # (b, 1024, t)
        logits = einsum('b t h, b h n t -> b n t', c, logits) / math.sqrt(self.n_head)
        soft_one_hot = F.softmax(logits, dim=1)

        dim = 1
        idx_N = soft_one_hot.max(dim, keepdim=True)[1]
        if only_index:
            return idx_N
        hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, idx_N, 1.0)
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

        z_q = einsum('b n t, n d -> b d t', one_hot, value)

        return logits, idx_N, z_q


class QuantizeInterpolatedEMAResetAttention(nn.Module):
    def __init__(self, V, C, args):
        super().__init__()
        self.V = V
        self.C = C
        self.mu = args.mu
        self.using_znorm = args.using_znorm
        self.reset_codebook()
        self.attn = Attention(hidden_dim=self.C, attn_dim=self.C, n_head=8, norm_type='rms_norm')
        
    def reset_codebook(self):
        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('code_sum', torch.zeros(self.V, self.C).cuda())
        self.register_buffer('codebook', torch.zeros(self.V, self.C).cuda())
        self.register_buffer('code_count', torch.ones(self.V, device=self.codebook.device))

    def _tile(self, x):
        V_x, C = x.shape
        if V_x < self.V:
            n_repeats = (self.V + V_x - 1) // V_x
            std = 0.01 / np.sqrt(C)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        # Preprocess
        x = self.merge_BT(x)
        out = self._tile(x)
        self.codebook = out[:self.V]
        self.code_sum = self.codebook.clone()
        self.initted = x.new_tensor([True])
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.V, code_idx.shape[0], device=code_idx.device)  # V, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # V
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        # Preprocess
        x = self.merge_BT(x)

        code_onehot = torch.zeros(self.V, x.shape[0], device=x.device)  # V, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # V, w
        code_count = code_onehot.sum(dim=-1)  # V

        out = self._tile(x)
        code_rand = out[:self.V]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, V
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # V

        usage = (self.code_count.view(self.V, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.V, self.C) / self.code_count.view(self.V, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def merge_BT(self, x):
        # BCT -> BTC -> [BT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x
    
    def dequantize(self, idx):
        value = self.attn.to_v(self.codebook)
        one_hot = F.one_hot(idx, num_classes=self.V).to(value.device).float()
        z_q = einsum('b t n, n d -> b d t', one_hot, value)

        return z_q
    
    def forward(self, z, q, phi, return_idx=False):
        B, width, T = z.shape

        B, C, T = z.shape
        # downsampling
        rest_NC = F.interpolate(z, size=(q), mode='area') if (q != T) else z
        logits, code_idx, z_hat = self.attn(rest_NC, self.codebook)
        
        # We have observed that employing unsampled data for look-up and
        # appling EMA can lead to an increase in perplexity!
        z_tmp = z

        with torch.no_grad():
            code_org = self.attn(z_tmp, self.codebook, only_index=True)

        # quant_loss = torch.mean((z_hat - rest_NC)**2) + torch.mean((z_hat_2.detach()-rest_NC)**2) + \
        #             torch.mean((z_hat_2 - rest_NC.detach()) ** 2)
        
        z_hat = F.interpolate(z_hat, size=(T), mode='linear').contiguous()
        z_hat = phi(z_hat)
        # z_hat_2 = F.interpolate(z_hat_2, size=(T), mode='linear').contiguous()
        # z_hat_2 = phi(z_hat_2)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(z, code_org.view(-1))
        else : 
            perplexity = self.compute_perplexity(code_org.view(-1))        

        if return_idx:
            return z_hat, code_idx.reshape(B, q), perplexity
        
        return z_hat, perplexity
    

class QuantizeInterpolatedEmbedding(nn.Module):
    def __init__(self, V, C, args):
        super().__init__()
        self.V = V
        self.C = C
        self.using_znorm = args.using_znorm
        self.codebook = nn.Embedding(self.V, self.C)
        self.register_buffer('initted', torch.Tensor([True]))

    def init_codebook(self, x):
        pass
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.V, code_idx.shape[0], device=code_idx.device)  # V, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # V
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def quantize(self, z, q):
        B, C, T = z.shape
        # downsampling
        rest_NC = F.interpolate(z, size=(q), mode='area').permute(0, 2, 1).reshape(-1, C) if (q != T) else z.permute(0, 2, 1).reshape(-1, C)
        if self.using_znorm:
            rest_NC = F.normalize(rest_NC, dim=-1)
            sim = rest_NC @ F.normalize(self.codebook.weight.data.T, dim=0)
            idx_Bpn = torch.argmax(sim, dim=1).view(B, q)
        else:
            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.codebook.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(rest_NC, self.codebook.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_Bpn = torch.argmin(d_no_grad, dim=1).view(B, q)
        
        # We have observed that employing unsampled data for look-up and
        # appling EMA can lead to an increase in perplexity!
        z_tmp = z.permute(0, 2, 1).reshape(-1, C)
        if self.using_znorm:
            rest_NC = F.normalize(z_tmp, dim=-1)
            sim = rest_NC @ F.normalize(self.codebook.weight.data.T, dim=0)
            idx_org = torch.argmax(sim, dim=1).view(-1)
        else:
            d_no_grad = torch.sum(z_tmp, dim=1, keepdim=True) + torch.sum(self.codebook.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_tmp, self.codebook.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_org = torch.argmin(d_no_grad, dim=1).view(-1)

        return idx_Bpn, idx_org
        
    def dequantize(self, code_idx, T, phi):
        z_hat = self.codebook(code_idx).permute(0, 2, 1).contiguous()


        # unsampling
        if code_idx.shape[1] != T:
            z_hat = F.interpolate(z_hat, size=(T), mode='linear').contiguous()

        z_hat = phi(z_hat)
        return z_hat
    
    def forward(self, z, q, phi, return_idx=False):
        B, width, T = z.shape

        code_idx, code_org = self.quantize(z, q)
        z_hat = self.dequantize(code_idx, T, phi)

        # Update embeddings
        perplexity = self.compute_perplexity(code_org)        

        if return_idx:
            return z_hat, code_idx, perplexity
        
        return z_hat, perplexity