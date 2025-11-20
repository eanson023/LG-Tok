import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader, RandomSampler
import numpy as np

from options.tok_option import arg_parse
from data.t2m_dataset import MotionDataset

from models.tokenizer.tokenizer import Tokenizer
from models.tokenizer.tok_trainer import VQTokenizerTrainer
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from utils import paramUtil
from utils.get_opt import get_opt
from utils.evaluators import Evaluators
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric

# run faster
torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

os.environ["OMP_NUM_THREADS"] = "1"

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == "__main__":
    opt = arg_parse(True)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.text_emb_dir = pjoin(opt.data_root, 'text_embeddings')
        opt.joints_num = 22
        opt.min_motion_len = 40
        dim_pose = 67
        fps = 20
        radius = 4
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.text_emb_dir = pjoin(opt.data_root, 'text_embeddings')
        opt.joints_num = 21
        opt.min_motion_len = 24
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
    elif opt.dataset_name == "motionx":
        opt.data_root = './dataset/Motion-X/'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.text_emb_dir = pjoin(opt.data_root, 'text_embeddings')
        opt.joints_num = 22
        opt.min_motion_len = 40
        dim_pose = 67
        fps = 20
        radius = 4
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/motionx/Comp_v6_KLD005/opt.txt'
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
    elif opt.dataset_name == "cmp":
        opt.data_root = './dataset/CMP/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.text_emb_dir = pjoin(opt.data_root, 'text_embeddings')
        opt.joints_num = 22
        opt.min_motion_len = 40
        dim_pose = 67
        fps = 20
        radius = 4
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/cmp/Comp_v6_KLD01/opt.txt'
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
    else:
        raise KeyError('Dataset Does not Exists')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = Evaluators(opt.dataset_name, opt.device)

    net = Tokenizer(opt,
                    opt.num_latent_tokens, 
                    dim_pose, 
                    opt.latent_dim,
                    opt.depth, 
                    opt.dropout,
                    opt.ff_size, 
                    opt.activation,
                    opt.patch_size,
                    opt.norm_first,
                    opt.norm,
                    opt.qk_norm,
                    opt.pos_embed,
                    opt.rope_base,
                    opt.enc_moiton_text_embed,
                    opt.dec_latent_text_embed,
                    opt.cond_drop_prob,
                    opt.mae_motion_drop,
                    opt.mae_motion_drop_max,
                    opt.tfg,
                    opt.mfg,
                    opt.code_dim, 
                    opt.text_model,
                    opt.text_max_len,
                    opt.quant_type,
                    opt.max_motion_length
                    )

    pc_vq = sum(param.numel() for param in net.parameters() if param.requires_grad)
    print(net)
    
    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    trainer = VQTokenizerTrainer(opt, vq_model=net)
    
    if opt.dataset_name == "motionx":
        train_dataset = MotionDataset(opt, mean, std, train_split_file, part=['short'])
        val_dataset = MotionDataset(opt, mean, std, val_split_file, part=['short'])
        train_dataset_ood = MotionDataset(opt, mean, std, train_split_file, part=['long'])
        batch_size = opt.batch_size // 2
        num_workers = 4
        samples = min(len(train_dataset_ood), len(train_dataset))
        train_loader_ood = DataLoader(train_dataset_ood, batch_size=batch_size, drop_last=True, num_workers=num_workers,
                                        pin_memory=True, sampler=RandomSampler(train_dataset_ood, num_samples=samples), persistent_workers=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers,
                                    pin_memory=True, sampler=RandomSampler(train_dataset, num_samples=samples), persistent_workers=True)
    else:
        train_dataset = MotionDataset(opt, mean, std, train_split_file)
        val_dataset = MotionDataset(opt, mean, std, val_split_file)
        batch_size = opt.batch_size
        num_workers = 8
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers,
                                    pin_memory=True, shuffle=True)
        train_loader_ood = None  # No OOD data for T2M, KIT, CMP datasets

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=num_workers,
                            shuffle=True, pin_memory=True)
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    trainer.train(train_loader, train_loader_ood, val_loader, eval_val_loader, eval_wrapper, plot_t2m)
