import os
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

from options.tok_option import arg_parse
from utils.tools import fixseed
from utils.get_opt import get_opt
import utils.eval_t2m as eval_t2m
from utils.evaluators import Evaluators
from models.tokenizer.tokenizer import Tokenizer
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

def load_vq_model(vq_opt, which_epoch):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')

    vq_model = Tokenizer(vq_opt,
                    vq_opt.num_latent_tokens, 
                    dim_pose, 
                    vq_opt.latent_dim,
                    vq_opt.depth, 
                    vq_opt.dropout,
                    vq_opt.ff_size, 
                    vq_opt.activation,
                    vq_opt.patch_size,
                    vq_opt.norm_first,
                    vq_opt.norm,
                    vq_opt.qk_norm,
                    vq_opt.pos_embed,
                    vq_opt.rope_base,
                    vq_opt.enc_moiton_text_embed,
                    vq_opt.dec_latent_text_embed,
                    vq_opt.cond_drop_prob,
                    vq_opt.mae_motion_drop,
                    vq_opt.mae_motion_drop_max,
                    vq_opt.tfg,
                    vq_opt.mfg,
                    vq_opt.code_dim, 
                    vq_opt.text_model,
                    vq_opt.text_max_len,
                    vq_opt.quant_type
                    )
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    # exclude text_model parameters
    ckpt[model_key] = {k: v for k, v in ckpt[model_key].items() if not k.startswith('text_model.')}
    missing_keys, unexpected_keys = vq_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('text_model.') for k in missing_keys])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch


if __name__ == "__main__":
    ##### ---- Exp dirs ---- #####
    args = arg_parse(False)
    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))

    fixseed(args.seed)

    args.out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(args.out_dir, exist_ok=True)

    f = open(pjoin(args.out_dir, '%s.log'%args.ext), 'w')

    if args.dataset_name == 'kit':
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
    elif args.dataset_name == 't2m':
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        data_root = './dataset/HumanML3D'
    elif args.dataset_name == 'motionx':
        dataset_opt_path = 'checkpoints/motionx/Comp_v6_KLD005/opt.txt'
        data_root = './dataset/Motion-X'
    elif args.dataset_name == 'cmp':
        dataset_opt_path = 'checkpoints/cmp/Comp_v6_KLD01/opt.txt'
        data_root = './dataset/CMP'
    else:
        raise NotImplementedError('please select a valid dataset!!')

    eval_wrapper = Evaluators(args.dataset_name, args.device)

    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))

    ##### ---- Dataloader ---- #####
    args.nb_joints = 21 if args.dataset_name == 'kit' else 22
    dim_pose = 63 if args.dataset_name == 'kit' else 67
    

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=args.device)

    ##### ---- Network ---- #####
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=args.device)
    vq_opt.min_motion_len = 24 if args.dataset_name == 'kit' else 40
    vq_opt.text_emb_dir = pjoin(data_root, 'text_embeddings')
    # net = load_vq_model()

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    for file in os.listdir(model_dir):
        # if not file.endswith('tar'):
        #     continue
        # if not file.startswith('net_best_fid'):
        #     continue
        if args.which_epoch != "all" and args.which_epoch not in file:
            continue
        print(file)
        net, ep = load_vq_model(vq_opt, file)

        net.eval()
        net.cuda()

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mae = []
        cb_usage = []
        repeat_time = 20
        for i in range(repeat_time):
            best_fid, best_div, Rprecision, best_matching, l1_dist, usage = \
                eval_t2m.evaluation_vqvae_plus_mpjpe_cbusage(eval_val_loader, net, i, eval_wrapper=eval_wrapper, num_joint=args.nb_joints, tfg=args.tfg, mfg=args.mfg, train_mean=mean, train_std=std)
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mae.append(l1_dist)
            cb_usage.append(usage.detach().cpu().numpy())

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mae = np.array(mae)
        cb_usage = np.array(cb_usage)

        print(f'{file} final result, epoch {ep}')
        print(f'{file} final result, epoch {ep}', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMAE:{np.mean(mae):.4f}, conf.{np.std(mae)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tCB Usage:{np.mean(cb_usage):.3f}, conf.{np.std(cb_usage)*1.96/np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()
