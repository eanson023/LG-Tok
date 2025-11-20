import torch
from functools import partial

    
def compute_cis(dim: int, end_x: int, theta: float = 100.0, normalize=32):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t_x = init_t_x(end_x, normalize=normalize)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    return freqs_cis_x

def init_t_x(end_x: int, normalize=-1):
    t_x = torch.arange(end_x, dtype=torch.float32)
    if normalize != -1 and t_x.shape[0] > 1:
        t_x = t_x / t_x.max() * normalize
    return t_x

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):  # L, C
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1]):  # B, L, C
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
        shape[0] = x.shape[0]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

if __name__ == '__main__':
    dtype = torch.float16
    embed_dim = 512
    num_heads = embed_dim // 64
    rope_theta = 100
    seq_lens = [16, 24, 32]
    values_list = []
    
    for seq_len in seq_lens:
        compute_cis = partial(compute_cis, dim=64, theta=rope_theta, normalize=32)
        freqs_cis = compute_cis(end_x=seq_len)
        
        q = torch.ones([1, num_heads, seq_len, 64]).to(dtype)
        k = torch.ones([1, num_heads, seq_len, 64]).to(dtype)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
        attn1 = q.float() @ k.float().transpose(-2, -1)
        attn1 = attn1.to(dtype)
        print(attn1.max(),attn1.min())
        print(attn1[0,0,0].max(),attn1[0,0,0].min())

        values_list.append(torch.diag(attn1[0, 0]).cpu().numpy())

