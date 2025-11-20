import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.tokenizer.quantizer import QuantizeInterpolatedEMAReset

# main class
class ScaleVQ(nn.Module):
    def __init__(
        self,
        scales,
        nb_code_st,
        nb_code_ed,
        code_dim, 
        args, 
        shared_codebook = False,
        phi_k = 3, 
        phi_depth = 2, 
    ):
        super().__init__()

        num_quantizers = len(scales)
        self.beta = args.commit_beta
        self.scales = scales
        self.code_dim = code_dim

        self.nb_codes = [round(nb_code) for nb_code in np.linspace(nb_code_st, nb_code_ed, num_quantizers)]

        if shared_codebook:
            assert nb_code_st == nb_code_ed
            layer = QuantizeInterpolatedEMAReset(nb_code_st, code_dim, args)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeInterpolatedEMAReset(self.nb_codes[i], code_dim, args) for i in range(num_quantizers)])

        self.phis = nn.ModuleList([Phi(code_dim, phi_k, phi_depth) for _ in range(len(self.scales))])
        
        # only used for progressive training
        self.prog_q = -1

    def forward(self, z_BCT):
        
        if z_BCT.dtype != torch.float32: z_BCT = z_BCT.float()
        B, C, T = z_BCT.shape
        z_no_grad = z_BCT.detach()
        
        z_q = z_no_grad.clone()
        z_hat = torch.zeros_like(z_q)

        all_losses = []
        all_usages = []
        all_indices = []
        all_perplexity = []

        for q, s_q in enumerate(self.scales):
            if 0 <= self.prog_q < q: break    # progressive training
            quantizer = self.layers[q]
            phi = self.phis[q]
            V = self.nb_codes[q]

            # Init codebook if not inited
            if self.training and not quantizer.initted:
                quantizer.init_codebook(z_BCT)

            z_hat_q, *rest = quantizer(z_q, s_q, phi, return_idx=True) #single quantizer

            # Fix bug: RuntimeError: one of the variables needed for gradient computation
            # has been modified by an inplace operation
            z_hat = z_hat + z_hat_q
            z_q = z_q - z_hat_q # Eq. (7)

            embed_indices, perplexity = rest
            all_indices.append(embed_indices)
            all_usages.append(z_q.new_tensor(embed_indices.unique().numel()/V * 100))
            # Calc loss 
            all_losses.append(F.mse_loss(z_hat.data, z_BCT).mul_(self.beta) + F.mse_loss(z_hat, z_no_grad))
            all_perplexity.append(perplexity)   

        # stack all losses and indices
        all_usages = sum(all_usages)/len(all_usages)
        all_losses = sum(all_losses)/len(all_losses)
        all_perplexity = sum(all_perplexity)/len(all_perplexity)

        # Passthrough
        z_hat = (z_hat.data - z_no_grad).add_(z_BCT)

        ret = (z_hat, all_usages, all_losses, all_perplexity, all_indices)
        return ret
    
    def quantize(self, z_BCT):
        
        if z_BCT.dtype != torch.float32: z_BCT = z_BCT.float()
        B, C, T = z_BCT.shape
        z_no_grad = z_BCT.detach()
        
        z_q = z_no_grad.clone()

        all_indices = []

        for q, s_q in enumerate(self.scales):
            if 0 <= self.prog_q < q: break    # progressive training

            quantizer = self.layers[q]
            phi = self.phis[q]

            z_hat_q, *rest = quantizer(z_q, s_q, phi, return_idx=True) #single quantizer

            z_q = z_q - z_hat_q

            embed_indices, _ = rest
            all_indices.append(embed_indices)
        
        return all_indices
    
    @torch.no_grad()
    def idxBT_to_t2m_input(self, gt_idx_BT, cat_res=True):
        next_scales = []
        
        scales = self.scales
        Q = len(gt_idx_BT)

        B = gt_idx_BT[0].shape[0]
        C = self.code_dim
        T = scales[-1]
        
        z_hat = gt_idx_BT[0].new_zeros(B, C, T, dtype=torch.float32)
        s_q_next = scales[0]
        for q in range(Q-1):
            if self.prog_q == 0 or (0 <= self.prog_q-1 < q): break   # progressive training
            quantizer = self.layers[q]
            # dequantize & upsampling
            z_hat_q = F.interpolate(F.embedding(gt_idx_BT[q], quantizer.codebook).transpose_(1, 2).view(B, C, s_q_next), size=(T), mode='linear')
            z_hat.add_(self.phis[q](z_hat_q))
            s_q_next = scales[q+1]
            # downsampling
            next_scales.append(F.interpolate(z_hat, size=(s_q_next), mode='area').view(B, C, -1).transpose(1, 2))
        
        if len(next_scales) ==0:
            return None
        return torch.cat(next_scales, dim=1) if cat_res else next_scales
    
    @torch.no_grad()
    def get_next_autoregressive_input(self, q, z_hat, z_BCs, scales=None):
        if scales == None:
            scales = self.scales
        T = scales[-1]
        Q = len(scales)

        if q != Q-1:
            # upsampling
            z_hat_q = self.phis[q](F.interpolate(z_BCs, size=(T), mode='linear'))     # conv after upsample
            z_hat.add_(z_hat_q)
            return z_hat, F.interpolate(z_hat, size=(scales[q+1]), mode='area').permute(0, 2, 1).contiguous()
        else:
            z_hat_q = self.phis[q](z_BCs)
            z_hat.add_(z_hat_q)
            return z_hat, z_hat.permute(0, 2, 1).contiguous()
    
    
class Phi(nn.Module):
    def __init__(self, embed_dim, ks, depth):
        super().__init__()
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2),
                nn.ReLU()
            )
            blocks.append(block)
        self.phi = nn.Sequential(*blocks)
    
    def forward(self, z_hat_BCT):
        return self.phi(z_hat_BCT)
