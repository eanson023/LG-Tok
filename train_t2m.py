import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.t2m.mosa.transformer import Transformer
from models.t2m.momask.transformer import MaskTransformer as TransformerMoMaskStage1
from models.t2m.momask.transformer import ResidualTransformer as TransformerMoMaskStage2
from models.t2m.momask.trainer import MaskTransformerTrainer as TransformerTrainerMoMaskStage1
from models.t2m.momask.trainer import ResidualTransformerTrainer as TransformerTrainerMoMaskStage2
from models.t2m.mosa.trainer import TransformerTrainer
from models.tokenizer.tokenizer import Tokenizer

from options.t2m_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.tools import fixseed
from utils.evaluators import Evaluators
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_opt.text_emb_dir = pjoin(vq_opt.data_root, 'text_embeddings')
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
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    ckpt[model_key] = {k: v for k, v in ckpt[model_key].items() if not k.startswith('text_model.')}
    missing_keys, unexpected_keys = vq_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('text_model.') for k in missing_keys])
    print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 67
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'kit': #TODO
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 63
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'motionx':
        opt.data_root = './dataset/Motion-X'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 67
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/motionx/Comp_v6_KLD005/opt.txt'
    elif opt.dataset_name == 'cmp':
        opt.data_root = './dataset/CMP/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 67
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/cmp/Comp_v6_KLD01/opt.txt'
    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')

    vq_model, vq_opt = load_vq_model()
    vq_model.eval()
    opt.model = vq_opt.quant_type

    clip_version = 'ViT-B/32'

    if opt.model == "mosa":
        t2m_transformer = Transformer(scales = vq_opt.scales,
                                        nb_code_st = vq_opt.nb_code_st,
                                        nb_code_ed = vq_opt.nb_code_ed,
                                        code_dim=vq_opt.code_dim,
                                        cond_mode='text',
                                        latent_dim=opt.latent_dim,
                                        ff_size=opt.ff_size,
                                        num_layers=opt.n_layers,
                                        num_heads=opt.n_heads,
                                        dropout=opt.dropout,
                                        clip_dim=512,
                                        cond_drop_prob=opt.cond_drop_prob,
                                        clip_version=clip_version,
                                        opt=opt)
        
        trainer = TransformerTrainer(opt, t2m_transformer, vq_model)
        print('Training MoSa t2m model')
    elif opt.model == "momask": 
        assert vq_opt.nb_code_st == vq_opt.nb_code_ed
        opt.num_tokens = vq_opt.nb_code_st
        opt.num_quantizers = vq_opt.quant_layers

        if opt.stage == 1:
            opt.model = "momask_stage1"
            t2m_transformer = TransformerMoMaskStage1(
                                                    code_dim=vq_opt.code_dim,
                                                    cond_mode='text',
                                                    latent_dim=opt.latent_dim,
                                                    ff_size=opt.ff_size,
                                                    num_layers=opt.n_layers,
                                                    num_heads=opt.n_heads,
                                                    dropout=opt.dropout,
                                                    cond_drop_prob=opt.cond_drop_prob,
                                                    clip_dim=512,
                                                    clip_version=clip_version,
                                                    opt=opt
                                                    )
            
            trainer = TransformerTrainerMoMaskStage1(opt, t2m_transformer, vq_model)
        if opt.stage == 2:
            opt.model = "momask_stage2"
            t2m_transformer = TransformerMoMaskStage2(
                                                    code_dim=vq_opt.code_dim,
                                                    cond_mode='text',
                                                    latent_dim=opt.latent_dim,
                                                    ff_size=opt.ff_size,
                                                    num_layers=opt.n_layers,
                                                    num_heads=opt.n_heads,
                                                    dropout=opt.dropout,
                                                    cond_drop_prob=opt.cond_drop_prob,
                                                    clip_dim=512,
                                                    clip_version=clip_version,
                                                    shared_codebook=vq_opt.shared_codebook,
                                                    share_weight=opt.share_weight,
                                                    opt=opt
                                                    )
            
            trainer = TransformerTrainerMoMaskStage2(opt, t2m_transformer, vq_model)
    else:
        raise NotImplementedError("")

    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters_wo_clip())
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')            

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True, prefetch_factor=2)

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = Evaluators(opt.dataset_name, opt.device)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)
