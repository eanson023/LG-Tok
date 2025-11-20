import copy
from typing import Optional, Any, Union, Callable, Tuple

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn import MultiheadAttention as OfficalMultiheadAttention

from .position_encoding import RotatoryPositionEmbedding1D, RotatoryPositionEmbedding2D

__all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer']



class RMSNorm(Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed
    


def _get_clone(module):
    return copy.deepcopy(module)

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class MultiheadAttention(Module):
    """_summary_

    Args:
        Module (_type_): _description_
    """

    def __init__(self, embed_dim=512, n_head=8, dropout=0.1, seq_len=100, rope='rope1d', qk_norm=False, rope_base=100.0,  **kwargs):
        super().__init__()
        assert embed_dim % 8 == 0
        
        self.use_rope = rope.startswith('rope')
        # key, query, value projections for all heads
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)

        if self.use_rope:
            self.rotary_embedding  = RotatoryPositionEmbedding1D(embed_dim=embed_dim//n_head, seq_len=seq_len, rope_base=rope_base)
            self.query_norm = RMSNorm(embed_dim) if qk_norm else nn.Identity()
            self.key_norm = RMSNorm(embed_dim) if qk_norm else nn.Identity()
            # if seq_len < 200:
            #     self.rotary_embedding  = RotatoryPositionEmbedding1D(embed_dim=embed_dim//n_head, seq_len=seq_len, rope_base=rope_base)
            # else:
            #     self.rotary_embedding  = RotatoryPositionEmbedding2D(embed_dim//n_head, rope_base, 77, 49, 196)
        else:
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
            self.rotary_embedding = nn.Identity()

        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)

        self.proj = Linear(embed_dim, embed_dim)

        self.n_head = n_head

        self.flash = True
        
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query.weight)
        xavier_uniform_(self.key.weight)
        xavier_uniform_(self.value.weight)
        constant_(self.proj.bias, 0.)

    def _prepare_attention_mask(self, attn_mask, key_padding_mask, q):
        B = q.size(0)
        combined_mask = None
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                combined_mask = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
                combined_mask.masked_fill_(attn_mask, float('-inf'))
            else:
                combined_mask = attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B, 1, 1, -1).expand(-1, self.n_head, -1, -1)
            
            if combined_mask is None:
                combined_mask = torch.zeros_like(key_padding_mask, dtype=q.dtype, device=q.device)
            
            if key_padding_mask.dtype == torch.bool:
                # Additive combination
                combined_mask = combined_mask.masked_fill(key_padding_mask, float('-inf'))
            else:
                combined_mask = combined_mask + key_padding_mask
        
        return combined_mask


    def forward(self, x, attn_mask=None, key_padding_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query_norm(self.query(x)).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key_norm(self.key(x)).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        q = self.rotary_embedding(q)                      
        k = self.rotary_embedding(k) 

        # efficient attention using Flash Attention CUDA kernels
        attn_mask = self._prepare_attention_mask(attn_mask, key_padding_mask, q)
        if self.flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is not None:
                att = att + attn_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class TransformerEncoder(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, latent_dim, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm_pre = norm
        self.norm_post = _get_clone(norm)

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2

        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*latent_dim, latent_dim), num_block)
    

    def forward(self, src, mask=None, src_key_padding_mask=None):

        x = src
        if self.norm_pre is not None:
            x = self.norm_pre(x)

        xs = []
        for mod in self.input_blocks:
            x = mod(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        for (mod, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = mod(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm_post is not None:
            x = self.norm_post(x)

        return x
    

class TransformerDecoder2(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, latent_dim, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm_pre = norm
        self.norm_post = _get_clone(norm)

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2

        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*latent_dim, latent_dim), num_block)

    def forward(self, tgt, memory, memory2, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                memory2_key_padding_mask = None,
                ):
        
        x = tgt
        if self.norm_pre is not None:
            x = self.norm_pre(x)

        xs = []
        for mod in self.input_blocks:
            x = mod(x, memory, memory2, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         memory2_key_padding_mask=memory2_key_padding_mask
                         )
            xs.append(x)

        x = self.middle_block(x, memory, memory2, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            memory2_key_padding_mask=memory2_key_padding_mask
                            )

        for (mod, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = mod(x, memory, memory2, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         memory2_key_padding_mask=memory2_key_padding_mask
                         )

        if self.norm_post is not None:
            x = self.norm_post(x)

        return x


class TransformerDecoder(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, latent_dim, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm_pre = norm
        self.norm_post = _get_clone(norm)

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2

        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*latent_dim, latent_dim), num_block)

    def forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                ):
        
        x = tgt
        
        if self.norm_pre is not None:
            x = self.norm_pre(x)

        xs = []
        for mod in self.input_blocks:
            x = mod(x, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         )
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            )

        for (mod, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = mod(x, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         )

        if self.norm_post is not None:
            x = self.norm_post(x)

        return x


#  borrowed from PyTorch official implementation
class TransformerEncoderLayer(Module):

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, norm_type="LN", seq_len=196+49, rope='rope1d', qk_norm=False, rope_base=100,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, \
                                            seq_len=seq_len, rope=rope, qk_norm=qk_norm, rope_base=rope_base,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear1_swi = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if norm_type == "LN":
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        elif norm_type == "RMS":
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            raise NotImplementedError
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.linear2(self.dropout(F.silu(self.linear1_swi(x)) * self.linear1(x)))
        return self.dropout2(x)



class TransformerDecoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, norm_type="LN", seq_len=196, rope='rope1d', qk_norm=False, rope_base=100,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, \
                                            seq_len=seq_len, rope=rope, qk_norm=qk_norm, rope_base=rope_base,
                                            **factory_kwargs)
        self.multihead_attn = OfficalMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if norm_type == "LN":
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        elif norm_type == "RMS":
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm3 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            raise NotImplementedError
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)
    

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class TransformerDecoderLayer2(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, qk_norm=False, rope_base=100,
                 device=None, dtype=None, norm_type="LN", seq_len=196, rope='rope1d') -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, \
                                            seq_len=seq_len, rope=rope, qk_norm=qk_norm, rope_base=rope_base,
                                            **factory_kwargs)
        self.multihead_attn = OfficalMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.multihead_attn2 = OfficalMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear1_swi = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if norm_type == "LN":
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm4 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        elif norm_type == "RMS":
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm3 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm4 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            raise NotImplementedError
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory2: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory2_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._mha_block2(self.norm4(x), memory2, memory2_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm4(x + self._mha_block2(x, memory2, memory2_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)
    
    # multihead attention block
    def _mha_block2(self, x: Tensor, mem: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn2(x, mem, mem, key_padding_mask=key_padding_mask)[0]
        return self.dropout4(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.linear2(self.dropout(F.silu(self.linear1_swi(x)) * self.linear1(x)))
        return self.dropout3(x)


    

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerDecoderLayerAdaLN(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, norm_type="LN", seq_len=196) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, seq_len=seq_len,
                                            **factory_kwargs)
        self.multihead_attn = OfficalMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model, bias=True)
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if norm_type == "LN":
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm4 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        elif norm_type == "RMS":
            self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm3 = RMSNorm(d_model, eps=layer_norm_eps)
            self.norm4 = RMSNorm(d_model, eps=layer_norm_eps)
        else:
            raise NotImplementedError
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        c: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        shift_msa, scale_msa, gate_msa, shift_csa, scale_csa, gate_csa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = tgt
        if self.norm_first:
            x = x + gate_msa.unsqueeze(1) * self._sa_block(modulate(self.norm1(x), shift_msa, scale_msa), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + gate_csa.unsqueeze(1) * self._mha_block(modulate(self.norm2(x), shift_csa, scale_csa), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            # x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + gate_mlp.unsqueeze(1) * self._ff_block(modulate(self.norm3(x), shift_mlp, scale_mlp))
        else:
            x = self.norm1(x + gate_msa.unsqueeze(1) * self._sa_block(modulate(x, shift_msa, scale_msa), tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + gate_csa.unsqueeze(1) * self._mha_block(modulate(x, shift_csa, scale_csa), memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + gate_mlp.unsqueeze(1) * self._ff_block(modulate(x, shift_mlp, scale_mlp)))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)
    
    # multihead attention block
    def _mha_block2(self, x: Tensor, mem: Tensor) -> Tensor:
        x = self.multihead_attn2(x, mem, mem)[0]
        return self.dropout4(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
