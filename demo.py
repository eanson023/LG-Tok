import os
from os.path import join as pjoin

import time
import torch

from models.t2m.mosa.transformer import Transformer
from models.tokenizer.tokenizer import Tokenizer

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.tools import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

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
        t2m_transformer = Transformer(scales = model_opt.scales,
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
    else:
        raise NotImplementedError("")
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'ema_mardm'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    # fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 61 if opt.dataset_name == 'kit' else 67

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

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

    #######################
    ######Loading Tokenizer######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_opt.text_emb_dir = pjoin(data_root, 'text_embeddings')
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.scales = vq_opt.scales
    model_opt.nb_code_st = vq_opt.nb_code_st
    model_opt.nb_code_ed = vq_opt.nb_code_ed
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, 'net_best_fid.tar')

    t2m_transformer.eval()
    vq_model.eval()

    t2m_transformer.to(opt.device)
    vq_model.to(opt.device) 

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    if opt.text_prompt != "":
        prompt_list.append(opt.text_prompt)
        length_list.append(opt.motion_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    raise "Please input a motion length like 'a persion walk forward.#196'!!!"
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise "A text prompt, or a file a text prompts are required!!!"
    # print('loading checkpoint {}'.format(file))
    
    token_lens = torch.LongTensor(length_list)
    token_lens[token_lens>model_opt.max_motion_length] = model_opt.max_motion_length
    token_lens = token_lens // 4
    token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = kit_kinematic_chain if opt.dataset_name == 'kit' else t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    total_time = time.time()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            z_hat = t2m_transformer.generate(captions, m_length,
                                            cond_scale=opt.cond_scale,
                                            vq_model=vq_model,
                                            top_k=opt.top_k, 
                                            top_p=opt.top_p, 
                                            temperature=opt.temperature,
                                            more_smooth=True)
            
            pred_motions = vq_model.dequantize(z_hat, m_lens=m_length, texts=captions, tfg=opt.tfg)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)
        
        # data = np.load('dataset/Motion-X/vector_263/dance/subset_0002/Several_Methods_Of_Ballet_Idling_clip_5.npy')[None, :, :]

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.nb_joints).numpy()

            if opt.nb_joints == 21:
                save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
                plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20, radius=246 * 12)
            else:
                bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
                _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

                bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
                _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)


                save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
                ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

                plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
                plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
                np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
                np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)

    total_time = time.time() - total_time
    print(f'Average Inference Time: {total_time/opt.repeat_times:.5f}')

