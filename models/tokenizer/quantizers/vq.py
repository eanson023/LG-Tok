import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

import random
from random import randrange
from einops import rearrange, repeat, pack, unpack

from models.tokenizer.quantizers import QuantizeInterpolatedEMAReset, QuantizeEMAReset

class BaseVQ(nn.Module, ABC):
    """
    Base class for Vector Quantization models.
    Provides common functionality for multi-layer quantizers.
    """
    
    def __init__(self, num_quantizers, code_dim, shared_codebook=False):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.code_dim = code_dim
        self.shared_codebook = shared_codebook
        self.layers = None  # Should be initialized by subclasses
    
    @property
    def codebooks(self):
        """Return stacked codebooks from all layers"""
        if self.layers is None:
            raise NotImplementedError("layers must be initialized by subclass")
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks  # 'q c d'
    
    def get_codes_from_indices(self, indices):
        """
        Get codes from indices for dequantization
        Args:
            indices: shape 'b n q' or list of indices
        Returns:
            all_codes: shape 'q b n d'
        """
        if isinstance(indices, list):
            # Handle list of indices (for ScaleVQ)
            batch = indices[0].shape[0]
            all_codes = []
            for q, idx in enumerate(indices):
                codes = F.embedding(idx, self.layers[q].codebook)
                all_codes.append(codes)
            return all_codes
        else:
            # Handle stacked indices (for ResidualVQ)
            batch, quantize_dim = indices.shape[0], indices.shape[-1]
            
            if quantize_dim < self.num_quantizers:
                indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
            
            from einops import repeat
            codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)
            gather_indices = repeat(indices, 'b n q -> q b n d', d=codebooks.shape[-1])
            
            mask = gather_indices == -1.
            gather_indices = gather_indices.masked_fill(mask, 0)
            
            all_codes = codebooks.gather(2, gather_indices)
            all_codes = all_codes.masked_fill(mask, 0.)
            
            return all_codes  # 'q b n d'
    
    def get_codebook_entry(self, indices):
        """
        Get codebook entries from indices
        Args:
            indices: indices for lookup
        Returns:
            latent: reconstructed features
        """
        if isinstance(indices, list):
            # For ScaleVQ - implementation depends on specific requirements
            raise NotImplementedError("get_codebook_entry for list indices should be implemented by subclass")
        else:
            # For ResidualVQ
            all_codes = self.get_codes_from_indices(indices)  # 'q b n d'
            latent = torch.sum(all_codes, dim=0)  # 'b n d'
            latent = latent.permute(0, 2, 1)
            return latent
    
    @abstractmethod
    def forward(self, x, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def quantize(self, x, **kwargs):
        """Quantization method - must be implemented by subclasses"""
        pass
    
    def _create_quantizer_layers(self, quantizer_class, quantizer_params_list):
        """
        Helper method to create quantizer layers
        Args:
            quantizer_class: Class to instantiate
            quantizer_params_list: List of parameters for each quantizer
        """
        if self.shared_codebook:
            # Use same parameters for all layers
            layer = quantizer_class(**quantizer_params_list[0])
            self.layers = nn.ModuleList([layer for _ in range(self.num_quantizers)])
        else:
            # Create different layers with different parameters
            self.layers = nn.ModuleList([
                quantizer_class(**params) for params in quantizer_params_list
            ])


class ScaleVQ(BaseVQ):
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
        num_quantizers = len(scales)
        super().__init__(num_quantizers, code_dim, shared_codebook)

        self.beta = args.commit_beta
        self.scales = scales

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


class ResidualVQ(BaseVQ):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        nb_code,
        code_dim,
        args,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        sample_codebook_temp=0.0,
        **kwargs
    ):
        super().__init__(num_quantizers, code_dim, shared_codebook)

        self.V = nb_code
        self.beta = args.commit_beta

        if shared_codebook:
            layer = QuantizeEMAReset(nb_code, code_dim, args)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(nb_code, code_dim, args) for _ in range(num_quantizers)])

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

        self.sample_codebook_temp = sample_codebook_temp

            
    def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize
        """Override base implementation for ResidualVQ specific behavior"""
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        # print(gather_indices.max(), gather_indices.min())
        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # 'q b n d'

    def get_codebook_entry(self, indices): #indices shape 'b n q'
        """Override base implementation for ResidualVQ specific behavior"""
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes = False, force_dropout_index=-1):
        # debug check
        # print(self.codebooks[:,0,0].detach().cpu().numpy())
        num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_usages = []
        all_indices = []
        all_perplexity = []


        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        # To ensure the first-k layers learn things as much as possible, we randomly dropout the last q - k layers
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) # keep quant layers <= quantize_dropout_cutoff_index, TODO vary in batch
            null_indices_shape = [x.shape[0], x.shape[-1]] # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            # null_loss = 0.

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]]  # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        # print(force_dropout_index)
        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                # all_losses.append(null_loss)
                continue

            if self.training and not layer.initted:
                layer.init_codebook(x)

            # layer_indices = None
            # if return_loss:
            #     layer_indices = indices[..., quantizer_index] #gt indices

            # quantized, *rest = layer(residual, indices = layer_indices, sample_codebook_temp = sample_codebook_temp) #single quantizer TODO
            quantized, *rest = layer(residual, return_idx=True, temperature=self.sample_codebook_temp) #single quantizer

            loss = F.mse_loss(residual, quantized.detach()) + self.beta * \
               torch.mean((quantized - residual.detach())**2)

            # Passthrough
            quantized = residual + (quantized - residual).detach()

            # print(quantized.shape, residual.shape)
            residual -= quantized.detach()
            quantized_out += quantized

            embed_indices, perplexity = rest
            all_indices.append(embed_indices)
            all_usages.append(quantized.new_tensor(embed_indices.unique().numel()/self.V * 100))
            all_losses.append(loss)
            all_perplexity.append(perplexity)


        # stack all losses and indices
        all_indices = torch.stack(all_indices, dim=-1)
        all_usages = sum(all_usages)/len(all_usages)
        all_losses = sum(all_losses)/len(all_losses)
        all_perplexity = sum(all_perplexity)/len(all_perplexity)

        ret = (quantized_out, all_usages, all_losses, all_perplexity, all_indices)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret
    
    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):

            quantized, *rest = layer(residual, return_idx=True) #single quantizer

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, perplexity = rest
            all_indices.append(embed_indices)
            # print(quantizer_index, embed_indices[0])
            # print(quantizer_index, quantized[0])
            # break
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx
    
class IdentityVQ(nn.Module):
    """without vq, for autoencoder"""
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ret = (x, x.new_zeros(1), x.new_zeros(1), x.new_zeros(1), [])
        return ret
    
    def quantize(self, x, return_latent=False):
        return x



# class QuantizeEMAReset(nn.Module):
#     def __init__(self, nb_code, code_dim, args):
#         super().__init__()
#         self.nb_code = nb_code
#         self.code_dim = code_dim
#         self.mu = args.mu
#         self.reset_codebook()
        
#     def reset_codebook(self):
#         self.init = False
#         self.code_sum = None
#         self.code_count = None
#         self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

#     def _tile(self, x):
#         nb_code_x, code_dim = x.shape
#         if nb_code_x < self.nb_code:
#             n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
#             std = 0.01 / np.sqrt(code_dim)
#             out = x.repeat(n_repeats, 1)
#             out = out + torch.randn_like(out) * std
#         else :
#             out = x
#         return out

#     def init_codebook(self, x):
#         out = self._tile(x)
#         self.codebook = out[:self.nb_code]
#         self.code_sum = self.codebook.clone()
#         self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
#         self.init = True
        
#     @torch.no_grad()
#     def compute_perplexity(self, code_idx) : 
#         # Calculate new centres
#         code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
#         code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

#         code_count = code_onehot.sum(dim=-1)  # nb_code
#         prob = code_count / torch.sum(code_count)  
#         perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
#         return perplexity
    
#     @torch.no_grad()
#     def update_codebook(self, x, code_idx):
        
#         code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
#         code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

#         code_sum = torch.matmul(code_onehot, x)  # nb_code, w
#         code_count = code_onehot.sum(dim=-1)  # nb_code

#         out = self._tile(x)
#         code_rand = out[:self.nb_code]

#         # Update centres
#         self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
#         self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

#         usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
#         code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

#         self.codebook = usage * code_update + (1 - usage) * code_rand
#         prob = code_count / torch.sum(code_count)  
#         perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
#         return perplexity

#     def preprocess(self, x):
#         # NCT -> NTC -> [NT, C]
#         x = x.permute(0, 2, 1).contiguous()
#         x = x.view(-1, x.shape[-1])  
#         return x

#     def quantize(self, x):
#         # Calculate latent code x_l
#         k_w = self.codebook.t()
#         distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
#                                                                                             keepdim=True)  # (N * L, b)
#         _, code_idx = torch.min(distance, dim=-1)
#         return code_idx

#     def dequantize(self, code_idx):
#         x = F.embedding(code_idx, self.codebook)
#         return x

    
#     def forward(self, x):
#         N, width, T = x.shape

#         # Preprocess
#         x = self.preprocess(x)

#         # Init codebook if not inited
#         if self.training and not self.init:
#             self.init_codebook(x)

#         # quantize and dequantize through bottleneck
#         code_idx = self.quantize(x)
#         x_d = self.dequantize(code_idx)

#         # Update embeddings
#         if self.training:
#             perplexity = self.update_codebook(x, code_idx)
#         else : 
#             perplexity = self.compute_perplexity(code_idx)
        
#         # Loss
#         commit_loss = F.mse_loss(x, x_d.detach())

#         # Passthrough
#         x_d = x + (x_d - x).detach()

#         # Postprocess
#         x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
#         return x_d, x_d.new_tensor(0), commit_loss, perplexity, code_idx
