import math

import torch
from torch import nn

class LearablePositionEmbedding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        scale = embed_dim ** -0.5
        self.pos_embed = nn.Parameter(scale * torch.randn(seq_len, embed_dim))
    
    def forward(self, x):
        x = x + self.pos_embed[:x.size(1)].to(x.dtype)

        return x
        

"""
The 1D version RoPE is modified on https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer 
"""
class RotatoryPositionEmbedding1D(nn.Module):
    def __init__(self, embed_dim, seq_len, rope_base):
        super().__init__()
        self.embed_dim = embed_dim  
        self.base = rope_base
    
        x_sin, x_cos = self.build_rope(seq_len)                                   # 1, S, 1, E    ,    1, S, 1, E
        self.register_buffer("x_cos", x_cos)                                      # Register_buffer for easy switching of device
        self.register_buffer("x_sin", x_sin)                                      # Register_buffer for easy switching of device

    def build_rope(self, seq_len):
        '''
        Create theta as per the equation in the RoPe paper: theta = base ^ -2(i-1)/d for i belongs to [1, 2, ... d/2].  
        '''
        sequence   = torch.arange(seq_len).float().unsqueeze(-1)
        thetas     = - torch.arange(start=0, end=self.embed_dim, step=2).float() / self.embed_dim       # E
        thetas     = torch.repeat_interleave(thetas, 2, 0)                                      # E//2
        thetas     = torch.pow(self.base, thetas)                                                     # E//2
        values     = sequence * thetas                                                          # S, 1 * E//2 -> S, E//2
        cos_values = torch.cos(values).unsqueeze(0).unsqueeze(0)                                # S, E//2     -> 1, 1, S, E//2      Precompute and store cos values
        sin_values = torch.sin(values).unsqueeze(0).unsqueeze(0)                                # S, E     --> 1, S, 1, E      Precompute and store sin values
        return sin_values, cos_values      


    def forward(self, x):
        x1 = x * self.x_cos[:, :, :x.shape[2], :]                              # B, S, H, E  *  1, S, 1, E          -->  B, S, H, E            Multiply with its cos factor
        x_shifted = torch.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), -1)      # B, S, H, E//2 stack B, S, H, E//2  -->  B, S, H, E//2, 2      Shift values for sin multiplications
        x_shifted = x_shifted.reshape(x.shape)                                 # B, S, H, E//2, 2                   -->  B, S, H, E            Reshape to the original size
        x2 = x_shifted * self.x_sin[:, :, :x.shape[2], :]                      # B, S, H, E  *  1, S, 1, E          -->  B, S, H, E            Multiply x with its sin factor
        x = x1 + x2                                              # B, H, S, E//2  cat  B, H, S, E//2 -> B, H, S, E          Combine x and y rotational projections
        return x


class RotatoryPositionEmbedding2D(nn.Module):
    def __init__(self, embed_dim, seq_len, rope_base):
        super().__init__()
        self.embed_dim = embed_dim // 2     # Split the dimensions into two parts. We will use 1 part for x-axis and the other part for y-axis
        self.base = rope_base

        x_positions, y_positions = self.build_modal_positions(seq_len)

        x_sin, x_cos = self.generate_rope1D(x_positions)
        y_sin, y_cos = self.generate_rope1D(y_positions)

        self.register_buffer("x_cos", x_cos)
        self.register_buffer("x_sin", x_sin)
        self.register_buffer("y_cos", y_cos)
        self.register_buffer("y_sin", y_sin)

    def build_modal_positions(self, seq_len):
        """
        Construct x/y position tensors for all modalities.
        x: intra-modal index (0 to length-1)
        y: modality ID (0 for text, 1 for latent, 2 for motion)
        """
        all_x = []
        all_y = []
        start_idx = 0
        # max_len = max(modal_lengths)

        # text_length = modal_lengths[0]
        # latent_length = modal_lengths[1]
        # motion_length = modal_lengths[2]

        # 这是v3
        # text modal
        # all_x.append(torch.arange(text_length).long())
        # all_y.append(torch.arange(text_length).long())

        
        # for y_idx, length in enumerate(modal_lengths):
        #     x_pos = torch.arange(start_idx, start_idx + length).long()    # global x idx if needed
        #     y_pos = torch.full((length,), y_idx, dtype=torch.long)        # fill with modality id

        #     all_x.append(x_pos)
        #     all_y.append(y_pos)
        #     start_idx += length


        # 这是矩形
        def find_next_square(num):
            root = math.isqrt(num)
            if root * root == num:
                return root
            else:
                return root + 1
        
        # len = sum(modal_lengths)
        patch = find_next_square(seq_len) 
        cnt = 0
        for i in range(patch):
            tmp = patch
            if cnt + patch > seq_len:
                tmp = seq_len - cnt
            xx = [i] * tmp
            yy = list(range(tmp))
            all_x.append(torch.tensor(xx).long())
            all_y.append(torch.tensor(yy).long())
            cnt += patch
        
        # text_patch = find_next_square(text_length)
        # cnt = 0
        # for i in range(text_patch):
        #     tmp = text_patch
        #     if cnt + text_patch > text_length:
        #         tmp = text_length - cnt
        #     xx = [i] * tmp
        #     yy = list(range(tmp))
        #     all_x.append(torch.tensor(xx).long())
        #     all_y.append(torch.tensor(yy).long())
        #     cnt += text_patch

        # latent_patch = int(latent_length ** 0.5)
        # for i in range(latent_patch):
        #     xx = [i] * latent_patch
        #     yy = list(range(latent_patch))
        #     all_x.append(torch.tensor(xx).long()+text_patch)
        #     all_y.append(torch.tensor(yy).long()+text_patch)

        # motion_patch = int(motion_length ** 0.5)
        # for i in range(motion_patch):
        #     xx = [i] * motion_patch
        #     yy = list(range(motion_patch))
        #     all_x.append(torch.tensor(xx).long()+text_patch+latent_patch)
        #     all_y.append(torch.tensor(yy).long()+text_patch+latent_patch)


        x_positions = torch.cat(all_x).reshape(-1, 1)  # N, 1
        y_positions = torch.cat(all_y).reshape(-1, 1)  # N, 1

        return x_positions, y_positions

    def generate_rope1D(self, sequence):
        '''
        Create theta as per the equation in the RoPe paper: theta = 10000 ^ -2(i-1)/d for i belongs to [1, 2, ... d/2].  
        Note this d/2 is different from previous x/y axis split.
        '''
        thetas     = -2 * torch.arange(start=0, end=self.embed_dim//2) / self.embed_dim       # E//4
        thetas     = torch.repeat_interleave(thetas, 2, 0)                                      # E//2
        thetas     = torch.pow(self.base, thetas)                                                   # E//2
        values     = sequence * thetas                                                          # S, 1 * E//2 -> S, E//2
        cos_values = torch.cos(values).unsqueeze(0).unsqueeze(0)                                # S, E//2     -> 1, 1, S, E//2      Precompute and store cos values
        sin_values = torch.sin(values).unsqueeze(0).unsqueeze(0)                                # S, E//2     -> 1, 1, S, E//2      Precompute and store sin values
        return sin_values, cos_values       


    def forward(self, x):
        x_x = x[:, :, :, :self.embed_dim]                                            # B, H, S, E//2                                            Split half of the embeddings of x for x-axis
        x_y = x[:, :, :, self.embed_dim:]                                            # B, H, S, E//2                                            Split half of the embeddings of x for y-axis

        x_x1 = x_x * self.x_cos[:, :, :x.shape[2], :]                                                      # B, H, S, E//2  *  1, 1, S, E//2   ->  B, H, S, E//2      Multiply x-axis part of input with its cos factor as per the eq in RoPe
        x_x_shifted = torch.stack((-x_x[:, :, :, 1::2], x_x[:, :, :, ::2]), -1)      # B, H, S, E//2                     ->  B, H, S, E//4, 2   Shift values for sin multiplications are per the eq in RoPe
        x_x_shifted = x_x_shifted.reshape(x_x.shape)                                 # B, H, S, E//4, 2                  ->  B, H, S, E//2
        x_x2 = x_x_shifted * self.x_sin[:, :, :x.shape[2], :]                                              # B, H, S, E//2  *  1, 1, S, E//2   ->  B, S, E//2         Multiply x-axis part of x with its sin factor as per the eq in RoPe
        x_x = x_x1 + x_x2                                                            # Add sin and cosine value
        
        x_y1 = x_y * self.y_cos[:, :, :x.shape[2], :]                                                      # B, H, S, E//2  *  1, 1, S, E//2   ->  B, H, S, E//2      Multiply y-axis part of input with its cos factor as per the eq in RoPe
        x_y_shifted = torch.stack((-x_y[:, :, :, 1::2], x_y[:, :, :, ::2]), -1)      # B, H, S, E//2                     ->  B, H, S, E//4, 2   Shift values for sin multiplications are per the eq in RoPe
        x_y_shifted = x_y_shifted.reshape(x_y.shape)                                 # B, H, S, E//4, 2                  ->  B, H, S, E//2
        x_y2 = x_y_shifted * self.y_sin[:, :, :x.shape[2], :]                                              # B, H, S, E//2  *  1, 1, S, E//2   ->  B, H, S, E//2      Multiply y-axis part of x with its sin factor as per the eq in RoPe
        x_y = x_y1 + x_y2                                                            # Add sin and cosine value

        x = torch.cat((x_x, x_y), -1)                                                # B, H, S, E//2  cat  B, H, S, E//2 -> B, H, S, E          Combine x and y rotational projections
        return x