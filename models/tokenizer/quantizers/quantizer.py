import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    dim = -1,
    training = True
):

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind


class QuantizeEMAReset(nn.Module):
    def __init__(self, V, C, args):
        super(QuantizeEMAReset, self).__init__()
        self.V = V
        self.C = C
        self.mu = args.mu
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

    def quantize(self, x, sample_codebook_temp=0.):
        x = self.merge_BT(x)
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(-distance, dim = -1, temperature = sample_codebook_temp, stochastic=True, training = self.training)

        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.V, code_idx.shape[0], device=code_idx.device)  # V, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        # Preprocess
        x = self.merge_BT(x)

        code_onehot = torch.zeros(self.V, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        out = self._tile(x)
        code_rand = out[:self.V]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.V, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.V, self.C) / self.code_count.view(self.V, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand


        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def merge_BT(self, x):
        # BCT -> BTC -> [BT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x


    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        # print(code_idx[0])
        if return_idx:
            return x_d, code_idx, perplexity
        return x_d, perplexity
    


class QuantizeInterpolatedEMAReset(QuantizeEMAReset):
    def __init__(self, V, C, args):
        super().__init__(V, C, args)
        self.using_znorm = args.using_znorm
        
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

        # We have observed that employing unsampled data for EMA can lead to an increase in fast convergence!
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

