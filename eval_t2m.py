import os
from os.path import join as pjoin

import time
import torch

from models.t2m.mosa.transformer import Transformer as TransformerMoSa
from models.t2m.momask.transformer import MaskTransformer as TransformerMoMaskStage1
from models.t2m.momask.transformer import ResidualTransformer as TransformerMoMaskStage2
from models.tokenizer.tokenizer import Tokenizer

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

import utils.eval_t2m as eval_t2m
from utils.tools import fixseed
from utils.evaluators import Evaluators

import numpy as np

def load_vq_model(vq_opt):
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
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    ckpt[model_key] = {k: v for k, v in ckpt[model_key].items() if not k.startswith('text_model.')}
    missing_keys, unexpected_keys = vq_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('text_model.') for k in missing_keys])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model):
    if model_opt.model == "mosa":
        t2m_transformer = TransformerMoSa(scales = model_opt.scales,
                                        nb_code_st = model_opt.nb_code_st,
                                        nb_code_ed = model_opt.nb_code_ed,
                                        code_dim=model_opt.code_dim,
                                        cond_mode='text',
                                        latent_dim=model_opt.latent_dim,
                                        ff_size=model_opt.ff_size,
                                        num_layers=model_opt.n_layers,
                                        num_heads=model_opt.n_heads,
                                        dropout=model_opt.dropout,
                                        clip_dim=512,
                                        cond_drop_prob=model_opt.cond_drop_prob,
                                        clip_version=clip_version,
                                        opt=model_opt)
        ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                          map_location='cpu')
        model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'ema_mardm'
        # print(ckpt.keys())
        missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
        return t2m_transformer
    elif model_opt.model == "momask":
        t2m_transformer_stage1 = TransformerMoMaskStage1(
                                                    code_dim=vq_opt.code_dim,
                                                    cond_mode='text',
                                                    latent_dim=model_opt.latent_dim,
                                                    ff_size=model_opt.ff_size,
                                                    num_layers=model_opt.n_layers,
                                                    num_heads=model_opt.n_heads,
                                                    dropout=model_opt.dropout,
                                                    cond_drop_prob=model_opt.cond_drop_prob,
                                                    clip_dim=512,
                                                    clip_version=clip_version,
                                                    opt=model_opt
                                                    )
        t2m_transformer_stage2 = TransformerMoMaskStage2(
                                                    code_dim=vq_opt.code_dim,
                                                    cond_mode='text',
                                                    latent_dim=model_opt.latent_dim,
                                                    ff_size=model_opt.ff_size,
                                                    num_layers=model_opt.n_layers,
                                                    num_heads=model_opt.n_heads,
                                                    dropout=model_opt.dropout,
                                                    cond_drop_prob=model_opt.cond_drop_prob,
                                                    clip_dim=512,
                                                    clip_version=clip_version,
                                                    shared_codebook=vq_opt.shared_codebook,
                                                    share_weight=model_opt.share_weight,
                                                    opt=model_opt
                                                    )
        
        print('Loading MoMask with share_weight:', model_opt.share_weight)
        
        ckpt1 = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                          map_location='cpu')
        model_key = 't2m_transformer' if 't2m_transformer' in ckpt1 else 'model'
        # print(ckpt.keys())
        missing_keys, unexpected_keys = t2m_transformer_stage1.load_state_dict(ckpt1[model_key], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        print(f'Loading Transformer Stage 1 {opt.name} from epoch {ckpt1["ep"]}!')

        ckpt2 = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.res_name, 'model', which_model),
                          map_location='cpu')
        model_key = 'res_transformer' if 'res_transformer' in ckpt2 else 't2m_transformer'
        # print(ckpt.keys())
        missing_keys, unexpected_keys = t2m_transformer_stage2.load_state_dict(ckpt2[model_key], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        print(f'Loading Transformer Stage 2 {opt.res_name} from epoch {ckpt2["ep"]}!')
        return t2m_transformer_stage1, t2m_transformer_stage2
    else:
        raise NotImplementedError(f"Model type {model_opt.model} not supported")

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 63 if opt.dataset_name == 'kit' else 67

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    # Load res_name for MoMask if needed
    if hasattr(opt, 'res_name') and opt.res_name:
        res_root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name)
        res_model_opt_path = pjoin(res_root_dir, 'opt.txt')
        res_model_opt = get_opt(res_model_opt_path, device=opt.device)
        model_opt.share_weight = res_model_opt.share_weight
        model_opt.res_name = opt.res_name

    if opt.dataset_name == 'kit':
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
    elif opt.dataset_name == 't2m':
        data_root = './dataset/HumanML3D'
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    elif opt.dataset_name == 'motionx':
        dataset_opt_path = 'checkpoints/motionx/Comp_v6_KLD005/opt.txt'
        data_root = './dataset/Motion-X'
    else:
        raise NotImplementedError('please select a valid dataset!!')
    
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.text_emb_dir = pjoin(data_root, 'text_embeddings')
    vq_model, vq_opt = load_vq_model(vq_opt)
    # print(vq_model)

    model_opt.code_dim = vq_opt.code_dim
    model_opt.model = vq_opt.quant_type
    
    if model_opt.model == "mosa":
        model_opt.scales = vq_opt.scales
        model_opt.nb_code_st = vq_opt.nb_code_st
        model_opt.nb_code_ed = vq_opt.nb_code_ed
    elif model_opt.model == "momask":
        model_opt.num_tokens = vq_opt.nb_code_st
        model_opt.num_quantizers = vq_opt.quant_layers

    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = Evaluators(opt.dataset_name, opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    # model_dir = pjoin(opt.)
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        t2m_transformer = load_trans_model(model_opt, file)
        
        # Handle both single transformer and tuple of transformers (for MoMask)
        if isinstance(t2m_transformer, tuple):
            t2m_transformer_stage1, t2m_transformer_stage2 = t2m_transformer
            t2m_transformer_stage1.eval()
            t2m_transformer_stage2.eval()
            t2m_transformer_stage1.to(opt.device)
            t2m_transformer_stage2.to(opt.device)
            trans_for_eval = (t2m_transformer_stage1, t2m_transformer_stage2)
        else:
            t2m_transformer.eval()
            t2m_transformer.to(opt.device)
            trans_for_eval = t2m_transformer
        
        vq_model.eval()
        vq_model.to(opt.device)

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        clip = []
        mm = []


        repeat_time = 20
        for i in range(repeat_time):
            with torch.no_grad():
                best_fid, best_div, Rprecision, best_matching, best_clip, best_mm = \
                    eval_t2m.evaluation_transformer_test(eval_val_loader, vq_model, trans_for_eval,
                                                                       i, eval_wrapper=eval_wrapper,
                                                        cond_scale=opt.cond_scale,
                                                        tfg = opt.tfg,
                                                         top_k=opt.top_k, top_p=opt.top_p, temperature=opt.temperature,
                                                                       force_mask=opt.force_mask, cal_mm=True,
                                                        train_mean=mean, train_std=std, model=model_opt.model)
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            clip.append(best_clip)
            mm.append(best_mm)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        clip = np.array(clip)
        mm = np.array(mm)

        print(f'{file} final result:')
        print(f'{file} final result:', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tCLIP: {np.mean(clip):.3f}, conf. {np.std(clip) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()
