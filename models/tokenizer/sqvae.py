import torch
import torch.nn as nn

from models.tokenizer.encdec2 import Encoder, Decoder
from models.tokenizer.vq import ScaleVQ
from utils.tools import lengths_to_mask
import numpy as np

class SQVAE(nn.Module):
    def __init__(self,
                 args,
                #  clip_model,
                 input_width,
                 scales,
                 nb_code_st = 256,
                 nb_code_ed = 768,
                 code_dim = 512,
                 width = 256,
                 width_mul = (1, 2, 4),
                 depth = 2,
                 slot_group = 10,
                 dropout = 0.2
                 ):

        super().__init__()
        self.input_width = input_width
        self.code_dim = code_dim
        self.mu = args.mu
        self.group_num = slot_group
        self.T = args.max_motion_length
        self.slot_num = args.max_motion_length * slot_group

        self.encoder = Encoder(ch=width, ch_mult=(1, 2, 1), num_res_blocks=2, dropout=0.2, in_channels=input_width, z_channels=code_dim)
        self.decoder = Decoder(ch=width, ch_mult=(1, 2, 1), num_res_blocks=2, dropout=0.2, in_channels=input_width, z_channels=code_dim) 

        pqvae_config = {
            'scales': scales,
            'nb_code_st': nb_code_st,
            'nb_code_ed': nb_code_ed,
            'code_dim':code_dim, 
            'args': args,
            'shared_codebook': args.shared_codebook,
            'phi_k': args.phi_k,
            'phi_depth': args.phi_depth,
        }

        self.quantizer = ScaleVQ(**pqvae_config)

    def preprocess(self, x):
        # (bs, T, dim) -> (bs, dim, T)
        x = x.permute(0, 2, 1).float().contiguous()
        return x

    def quantize(self, m, m_lens, *kwargs):

        z, _ = self.forward_encoder(m, m_lens)

        code_idxs = self.quantizer.quantize(z)

        return code_idxs

    def forward(self, m, m_lens, *kwargs):
        mask = lengths_to_mask(m_lens, m.size(1))
        # Encode
        z, m_padded = self.forward_encoder(m, m_lens)
        
        z_hat, code_usage, commit_loss, perplexity, all_indices = self.quantizer(z)

        m_hat = self.decoder(z_hat)

        m_hat[~mask] = 0.

        return m_hat, code_usage, commit_loss, perplexity, all_indices

    def forward2quantize(self, m, m_lens, *kwargs):
        mask = lengths_to_mask(m_lens, m.size(1))
        # Encode
        z, m_padded = self.forward_encoder(m, m_lens)
        
        z_hat, code_usage, commit_loss, perplexity, all_indices = self.quantizer(z)

        

        return z_hat

    def forward_encoder(self, m, m_lens):
        m_in = self.preprocess(m)

        # if self.training and not self.initted:
        #     self.init_slot(m_in, m_lens)

        # # Aligning the maximum length of motions to enable batch sampling operations.
        # m_in = self.pad_slot(m_in, m_lens)
        # Encode
        z = self.encoder(m_in)
        
        return z, m_in.permute(0, 2, 1)

    def forward_decoder(self, z_hat):
        m_hat = self.decoder(z_hat)
        return m_hat
    
    def generation(self, z_hat, *kwargs):
        return self.forward_decoder(z_hat)
    
    def reconstruct(self,  m, m_lens, *kwargs):
        return self.forward(m, m_lens)
    
    def dequantize(self,  m, **kwargs):
        return self.forward_decoder(m)
