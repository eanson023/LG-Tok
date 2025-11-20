from data.t2m_dataset import Text2MotionDatasetEval, collate_fn
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

def get_dataset_motion_loader(opt_path, batch_size, fname, device):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name in ['t2m', 'kit', 'cmp']:
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(f'utils/eval_mean_std/{opt.dataset_name}/eval_mean.npy')
        std = np.load(f'utils/eval_mean_std/{opt.dataset_name}/eval_std.npy')

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    elif opt.dataset_name == 'motionx':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(f'utils/eval_mean_std/{opt.dataset_name}/eval_mean.npy')
        std = np.load(f'utils/eval_mean_std/{opt.dataset_name}/eval_std.npy')

        w_vectorizer = WordVectorizer('./glove', 'our_vab', 'glove_6B')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset