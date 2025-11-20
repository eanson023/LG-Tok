import torch
import torch.nn as nn
import scipy.stats as stats
import numpy as np

from models.tokenizer.network import TransformerEncoder, TransformerDecoder, TransformerDecoder2, TransformerEncoderLayer, TransformerDecoderLayer2, TransformerDecoderLayer, RMSNorm
from models.tokenizer.position_encoding import LearablePositionEmbedding

class Encoder(nn.Module):
    def __init__(
        self, 
        num_latent_tokens, 
        text_length, 
        enc_moiton_text_embed, 
        *, 
        ch=128, 
        patch_size=8, 
        num_layers=2,
        dropout=0.0, 
        in_channels=3, 
        z_channels, 
        num_heads = 4, 
        ff_size = 1024, 
        activation = 'gelu',
        norm_first = False,
        max_motion_length = 196, 
        norm = 'RMS', 
        qk_norm = False,
        pos_embed='rope1d', 
        rope_base=100,
        token_drop = 0.4, 
        token_drop_max = 0.6,
    ):
        super().__init__()
        
        self.ch = ch
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.max_motion_length = max_motion_length
        self.num_latent_tokens = num_latent_tokens
        self.text_length = text_length
        self.motion_text_embed = enc_moiton_text_embed
        
        self.to_patch_embedding = nn.Conv1d(in_channels, self.ch, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
       
        if norm == "RMS":
            post_norm = RMSNorm(self.ch)
        elif norm == "LN":
            post_norm = nn.LayerNorm(self.ch)
        else:
            raise NotImplementedError()
        
        if self.motion_text_embed == "ctx_ctx":
            seq_len = self.max_motion_length//self.patch_size+self.num_latent_tokens +self.text_length
            self.transformer = TransformerEncoder(TransformerEncoderLayer(
                                                    self.ch,
                                                    num_heads,
                                                    ff_size,
                                                    dropout,
                                                    activation,
                                                    batch_first=True,
                                                    seq_len=seq_len,
                                                    norm_type= norm,
                                                    norm_first=norm_first,
                                                    rope=pos_embed,
                                                    rope_base=rope_base,
                                                    qk_norm=qk_norm,
                                                ), num_layers, self.ch, norm=post_norm)
        elif self.motion_text_embed in ["ctx_crs", "crs_ctx"]:
            seq_len = self.num_latent_tokens
            if self.motion_text_embed.split('_')[0]=='ctx':
                seq_len += self.max_motion_length//self.patch_size
            else:
                seq_len += self.text_length
            self.transformer = TransformerDecoder(TransformerDecoderLayer(
                                                    self.ch,
                                                    num_heads,
                                                    ff_size,
                                                    dropout,
                                                    activation,
                                                    batch_first=True,
                                                    seq_len=seq_len,
                                                    norm_type= norm,
                                                    norm_first=norm_first,
                                                    rope=pos_embed,
                                                    rope_base=rope_base,
                                                    qk_norm=qk_norm,
                                                ), num_layers, self.ch, norm=post_norm)
        elif self.motion_text_embed == "crs_crs":
            seq_len = self.num_latent_tokens
            self.transformer = TransformerDecoder2(TransformerDecoderLayer2(
                                                    self.ch,
                                                    num_heads,
                                                    ff_size,
                                                    dropout,
                                                    activation,
                                                    batch_first=True,
                                                    seq_len = seq_len,
                                                    norm_type= norm,
                                                    norm_first=norm_first,
                                                    rope=pos_embed,
                                                    rope_base=rope_base,
                                                    qk_norm=qk_norm,
                                                ), num_layers, self.ch, norm=post_norm)
        else: raise NotImplementedError("")

        self.en_position_embedding = LearablePositionEmbedding(self.ch, seq_len) if not pos_embed.startswith('rope') else nn.Identity()
        
        self.conv_out = torch.nn.Conv1d(self.ch, z_channels, kernel_size=3, stride=1, padding=1)

        # MAE
        # token drop
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            scale = self.ch ** -0.5
            self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.ch))
            # nn.init.normal_(self.mask_token, std=.02)


    def sample_orders(self, bsz, seq_len):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask
    
    def forward(self, m, l, t, x_mask, l_mask, t_mask, mfg_mask, mode='normal'):

        # downsampling
        h = self.to_patch_embedding(m.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.en_position_embedding(h)

        if (self.token_drop and self.training) or (self.token_drop and mode == 'masking'):
            orders = self.sample_orders(bsz=h.size(0), seq_len=h.size(1)).to(m.device)
            mask = self.random_masking(h, orders)
            mask = (mask * (1-mfg_mask)[:, None]).bool()
            x_mask = x_mask & ~mask
            h = torch.where(mask.unsqueeze(-1), self.mask_token, h)

        x_mask_p = x_mask[:, ::self.patch_size]

        if self.motion_text_embed == "ctx_ctx":
            # text + latent + motion
            z_mask = torch.cat([t_mask, l_mask, x_mask_p], dim=1)
            z = torch.cat([t, l, h], dim=1)
            z = self.transformer(z, src_key_padding_mask=~z_mask)
            z = z[:, self.text_length:self.text_length+self.num_latent_tokens]
        # if self.motion_text_embed == "ctx_ctx":
        #     # text + latent + motion
        #     z_mask = torch.cat([l_mask, x_mask_p], dim=1)
        #     z = torch.cat([l, h], dim=1)
        #     z = self.transformer(z, src_key_padding_mask=~z_mask)
        #     z = z[:, :self.num_latent_tokens]
        elif self.motion_text_embed == "ctx_crs":
            # using cross attention for text
            z_mask = torch.cat([l_mask, x_mask_p], dim=1)
            z = torch.cat([l, h], dim=1)
            z = self.transformer(tgt=z, memory=t, tgt_key_padding_mask=~z_mask, memory_key_padding_mask=~t_mask)
            z = z[:, :self.num_latent_tokens]
        elif self.motion_text_embed == "crs_ctx":
            # using cross attention for motion
            z_mask = torch.cat([t_mask, l_mask], dim=1)
            z = torch.cat([t, l], dim=1)
            z = self.transformer(tgt=z, memory=h, memory_key_padding_mask=~x_mask_p, tgt_key_padding_mask=~z_mask)
            z = z[:, self.text_length:self.text_length+self.num_latent_tokens]
        else:
            # double cross attention
            z = self.transformer(tgt=l, memory=h, memory2=t, memory_key_padding_mask=~x_mask_p, memory2_key_padding_mask=~t_mask)

        z = z.permute(0, 2, 1)
        z = self.conv_out(z)
        return z


class Decoder(nn.Module):
    def __init__(
        self, 
        dec_latent_text_embed,
        num_latent_tokens,
        *, 
        ch=128, 
        patch_size=8, 
        num_layers=2,
        dropout=0.0, 
        in_channels=3,  
        z_channels=256, 
        text_length=77,
        max_motion_length=196,
        num_heads=4, 
        ff_size=1024, 
        activation='gelu',
        norm_first=False,
        norm='RMS',
        qk_norm=False,
        pos_embed='rope1d',
        rope_base=100,
    ):
        super().__init__()
        
        self.ch = ch
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.num_latent_tokens = num_latent_tokens
        self.text_length = text_length
        self.max_motion_length = max_motion_length
        self.dec_latent_text_embed = dec_latent_text_embed
        
        self.conv_in_l = nn.Conv1d(z_channels, self.ch, kernel_size=3, stride=1, padding=1)

        if norm == "RMS":
            post_norm = RMSNorm(self.ch)
        elif norm == "LN":
            post_norm = nn.LayerNorm(self.ch)
        else:
            raise NotImplementedError(f"Unsupported norm type: {norm}")

        if self.dec_latent_text_embed == "ctx_ctx":
            seq_len = self.max_motion_length // self.patch_size + self.num_latent_tokens + self.text_length
            self.transformer = TransformerEncoder(TransformerEncoderLayer(
                                                    self.ch,
                                                    num_heads,
                                                    ff_size,
                                                    dropout,
                                                    activation,
                                                    batch_first=True,
                                                    seq_len=seq_len,
                                                    norm_first=norm_first,
                                                    norm_type= norm,
                                                    rope=pos_embed,
                                                    rope_base=rope_base,
                                                    qk_norm=qk_norm
                                                ), num_layers, self.ch, norm=post_norm)
        elif self.dec_latent_text_embed in ["ctx_crs", "crs_ctx"]:
            seq_len = self.max_motion_length // self.patch_size
            if self.dec_latent_text_embed.split('_')[0] == 'ctx':
                seq_len += self.num_latent_tokens
            else:
                seq_len += self.text_length
            self.transformer = TransformerDecoder(TransformerDecoderLayer(
                                                    self.ch,
                                                    num_heads,
                                                    ff_size,
                                                    dropout,
                                                    activation,
                                                    batch_first=True,
                                                    seq_len=seq_len,
                                                    norm_first=norm_first,
                                                    norm_type= norm,
                                                    rope=pos_embed,
                                                    rope_base=rope_base,
                                                    qk_norm=qk_norm
                                                ), num_layers, self.ch, norm=post_norm)
        elif self.dec_latent_text_embed == "crs_crs":
            seq_len = self.max_motion_length // self.patch_size
            self.transformer = TransformerDecoder2(TransformerDecoderLayer2(
                                                    self.ch,
                                                    num_heads,
                                                    ff_size,
                                                    dropout,
                                                    activation,
                                                    batch_first=True,
                                                    seq_len = seq_len,
                                                    norm_first=norm_first,
                                                    norm_type= norm,
                                                    rope=pos_embed,
                                                    rope_base=rope_base,
                                                    qk_norm=qk_norm
                                                ), num_layers, self.ch, norm=post_norm)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.dec_latent_text_embed}")

        self.de_position_embedding = LearablePositionEmbedding(self.ch, self.max_motion_length // self.patch_size)
        
        self.conv_out = nn.Sequential(
            nn.ConvTranspose1d(self.ch, in_channels, kernel_size=self.patch_size, stride=self.patch_size)
        )

    def forward(self, x, l, t, x_mask, l_mask, t_mask):
        """
        x: motion token embedding (B, Lx, C)
        z: latent embedding (B, Cz, Lz)
        t: text embedding (B, Lt, C)
        """
        l = self.conv_in_l(l).permute(0, 2, 1)  # (B, Lz, C)
        h = self.de_position_embedding(x)

        # motion patch mask
        x_mask_p = x_mask[:, ::self.patch_size]

        if self.dec_latent_text_embed == "ctx_ctx":
            mask = torch.cat([t_mask, l_mask, x_mask_p], dim=1)
            seq = torch.cat([t, l, h], dim=1)
            h = self.transformer(seq, src_key_padding_mask=~mask)
            h = h[:, self.text_length + self.num_latent_tokens:]
        elif self.dec_latent_text_embed == "ctx_crs":
            # tgt = latent + motion, memory = text
            mask = torch.cat([l_mask, x_mask_p], dim=1)
            tgt = torch.cat([l, h], dim=1)
            h = self.transformer(tgt=tgt, memory=t, tgt_key_padding_mask=~mask, memory_key_padding_mask=~t_mask)
            h = h[:, self.num_latent_tokens:]
        elif self.dec_latent_text_embed == "crs_ctx":
            # tgt = text + motion, memory = latent
            mask = torch.cat([t_mask, x_mask_p], dim=1)
            tgt = torch.cat([t, h], dim=1)
            h = self.transformer(tgt=tgt, memory=l, tgt_key_padding_mask=~mask)
            h = h[:, self.text_length:]
        # elif self.dec_latent_text_embed == "crs_ctx":
        #     # tgt = text + motion, memory = latent
        #     mask = x_mask_p
        #     tgt = h
        #     h = self.transformer(tgt=tgt, memory=l, tgt_key_padding_mask=~mask)
        else:  # crs_crs
            h = self.transformer(tgt=h, memory=l, memory2=t, tgt_key_padding_mask=~x_mask_p, memory_key_padding_mask=~l_mask, memory2_key_padding_mask=~t_mask)

        h = self.conv_out(h.permute(0, 2, 1))
        return h.permute(0, 2, 1)



if __name__ == "__main__":
    
    import torch

    encoder = Encoder(ch=128, ch_mult=(1, 2, 4), num_res_blocks=1, dropout=0.0, in_channels=263, z_channels_in=256, z_channels=512)
    decoder = Decoder(ch=128, ch_mult=(1, 2, 4), num_res_blocks=1, dropout=0.0, in_channels=263, x_channels_in=256 ,z_channels=512)
    
    print(encoder)
    print(decoder)

    enco_param = sum(param.numel() for param in encoder.parameters())
    deco_param = sum(param.numel() for param in decoder.parameters())

    print('Total parameters of all models: {}M'.format((enco_param+deco_param)/1000_000))
    
    input_data = torch.randn((64, 192, 263))
    x_mask = torch.ones((64, 192)).bool()
    z = torch.randn((64, 49, 256))
    m = torch.randn((64, 48, 256))
    z_mask = torch.ones((64, 49)).bool()
    with torch.no_grad():
        z = encoder(input_data, z, x_mask, z_mask)
        print(z.shape)

        rec = decoder(m, z, x_mask, z_mask)
        print(rec.shape)