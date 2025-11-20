import argparse
import os
import torch

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit', 'motionx', 'cmp'], help='dataset directory')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')

    ## optimization
    parser.add_argument('--max_epoch', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[180], nargs="+", type=int, help="learning rate schedule (epoch)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--grad_clip', default=0.01, type=float, help='gradient clipping threshold (0 to disable)')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.0, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    
    # language-guided model
    parser.add_argument("--text_model", type=str, default="clip", help='language mdoel, choice: clip, t5-small, t5-base, flan-t5-base, bert-base-uncased')
    parser.add_argument("--text_max_len", type=int, default=77, help='text sequence max length')

    ## transformer architecture
    parser.add_argument("--depth", type=int, default=9, help="Number of Transformer layers in both encoder and decoder")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent token embedding dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--ff_size", type=int, default=1024, help="Hidden dimension size of the feedforward network")
    parser.add_argument("--activation", type=str, default='gelu', help="Activation function used in Transformer blocks")
    parser.add_argument("--patch_size", type=int, default=1, help="ViT patch size")
    parser.add_argument('--norm_first', action="store_true", help="pre-norm or post-norm for transformer")
    parser.add_argument("--norm", type=str, default='RMS', choices=["RMS", "LN"], help="Normalization type: RMSNorm or LayerNorm")
    parser.add_argument("--qk_norm", action="store_true", help="using qk-norm before RoPE")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--num_latent_tokens", type=int, default=49, help="Number of latent tokens for sequence representation")
    parser.add_argument("--pos_embed", type=str, default="rope1d", choices=['learn', 'rope1d', 'rope2d'], help="Positional encoding type: learnable, 1D RoPE, or 2D RoPE")
    parser.add_argument("--rope_base", type=float, default=100.0, help="RoPE base parameter")
    parser.add_argument("--enc_moiton_text_embed", type=str, default="ctx_ctx", help="Moiton and Text embed method in encoder: ctx for in-context learning, crs for cross-attention")
    parser.add_argument("--dec_latent_text_embed", type=str, default="crs_crs", help="Moiton and Text embed method in decoder: ctx for in-context learning, crs for cross-attention")

    # motion/text classifier-free guidance
    parser.add_argument("--cond_drop_prob", type=float, default=0.1, help="Condition dropout probability for motion text-free guidance")
    parser.add_argument("--mae_motion_drop", type=float, default=0.4, help="Minimum motion token dropout ratio for motion masked autoencoding")
    parser.add_argument("--mae_motion_drop_max", type=float, default=0.6, help="Maximum motion token dropout ratio for motion masked autoencoding")
    parser.add_argument("--tfg", type=float, default=1.0, help="Text-free guidance scale")
    parser.add_argument("--mfg", type=float, default=1.0, help="Motion-free guidance scale")


    # quantizer
    parser.add_argument("--code_dim", type=int, default=512, help="codebook dimension C")
    parser.add_argument('--commit_beta', type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument("--quant_type", type=str, default="mosa", choices=['momask', 'mosa'], help='')
    parser.add_argument("--quant_layers", type=int, default=10, help="number of quantization layers")

    # 1. special hyper paramaters for MoSa
    parser.add_argument("--scales", type=str, default="3_6_10_15_20_25_30_36_42_49", help="scalble quantizer scales, represents a predefined scheduler moving from coarse to fine")
    parser.add_argument("--nb_code_st", type=int, default=256, help="nb of embedding start")
    parser.add_argument("--nb_code_ed", type=int, default=768, help="nb of embedding end")
    parser.add_argument("--using_znorm", action="store_true", help=', transforming the Euclidean distance into cosine similarity')
    parser.add_argument('--phi_k', type=int, default=3, help='conv block phi kernel')
    parser.add_argument('--phi_depth', type=int, default=2, help='conv block depth')

    # 2. special hyper paramaters for MoMask
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')
    parser.add_argument('--sample_codebook_temp', type=float, default=0.5, help='sample codebook temperature')

    parser.add_argument('--tiny', action="store_true", help="training on small datasets, small model")

    ## other
    parser.add_argument('--name', type=str, default="svq_nq10_nc256_768_noshare_phik3_phidepth2", help='Name of this trial')
    parser.add_argument('--is_continue', action="store_true", help='Name of this trial')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_every', default=50, type=int, help='iter log frequency')
    parser.add_argument('--save_latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--save_every_e', default=48, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=1, type=int, help='save eval results every n epoch')
    parser.add_argument('--eval_every_i', default=2000, type=int, help='save eval results every n iters')
    parser.add_argument('--eval_start_e', type=int, default=200, help='Frequency of animating eval results, (epoch)')
    # parser.add_argument('--early_stop_e', default=5, type=int, help='early stopping epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

    parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')
    parser.add_argument('--ext', type=str, default='default', help='eval file prefix')
    parser.add_argument("--unit_length", type=int, default=4, help="")
 
    parser.add_argument("--seed", default=2025, type=int)

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    if opt.tiny:
        opt.depth = 3
        # autoencoder
        # opt.quant_type = 'mar'
        
    opt.scales = tuple(map(int, opt.scales.replace('-', '_').split('_')))

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    return opt