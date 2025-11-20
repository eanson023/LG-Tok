import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import codecs as cs
import json

torch.set_float32_matmul_precision('high')

# 添加项目根目录到路径以便导入模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.tokenizer.text_wrapper import T5ModelWrapper, CLIPModelWrapper, \
                                         BertModelWrapper, ModernBertModelWrapper, LlamaModelWrapper, QwenModelWrapper

def create_text_model(text_model_name, max_length):
    if 't5' in text_model_name.lower():
        return T5ModelWrapper(text_model_name, max_length=max_length).cuda()
    elif 'vit' in text_model_name.lower(): 
        return CLIPModelWrapper(text_model_name).cuda()
    elif 'modernbert' in text_model_name.lower():
        return ModernBertModelWrapper(text_model_name, max_length=max_length).cuda()
    elif 'bert' in text_model_name.lower():
        return BertModelWrapper(text_model_name, max_length=max_length).cuda()
    elif 'llama' in text_model_name.lower():
        return LlamaModelWrapper(text_model_name, max_length=max_length).cuda()
    elif 'qwen' in text_model_name.lower():
        return QwenModelWrapper(text_model_name, max_length=max_length).cuda()
    
    else:
        raise ValueError(f"Unsupported text model: {text_model_name}. Supported models are T5 and CLIP.")

def load_text_data(text_dir, name):
    """加载单个文件的文本数据"""
    text_data = []
    text_file = os.path.join(text_dir, name + '.txt')
    
    if not os.path.exists(text_file):
        return text_data
        
    with cs.open(text_file, 'r') as f:
        for line in f.readlines():
            # 处理不同数据集的文本格式
            if 'kit-ml' in text_dir.lower() or 'humanml3d' in text_dir.lower() or 'cmp' in text_dir.lower() or 'humanml' in name:
                line_split = line.strip().split('#')
                caption = line_split[0]
            else:  # Motion-X格式
                caption = line.strip()
            if caption.strip():
                text_data.append(caption.strip())
    
    return text_data

def precompute_embeddings_for_dataset(dataset_path, text_model_name='flan-t5-base', max_length=77, device='cuda:0'):
    """为指定数据集预计算文本embedding"""
    
    print(f"Processing dataset: {dataset_path}")
    text_dir = os.path.join(dataset_path, 'texts')
    
    if not os.path.exists(text_dir):
        print(f"Text directory not found: {text_dir}")
        return
    
    # 创建输出目录
    output_dir = os.path.join(dataset_path, 'text_embeddings')
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化文本模型
    print(f"Initializing text model: {text_model_name}")
    text_model = create_text_model(text_model_name, max_length).to(device)
    text_model.eval()
    
    text_files = []
    for root, dirs, files in os.walk(text_dir):
        for file in files:
            if file.endswith('.txt'):
                rel_path = os.path.relpath(os.path.join(root, file), text_dir)
                text_files.append(rel_path[:-4])
    
    if not text_files:
        print(f"No text files found in {text_dir}")
        return

    all_text_files_batched = np.array_split(text_files, 400)

    nb_tokens = []

    nb_tokens_so_far = 0
    big_tensor = []
    index = []
    all_texts = []

    t_lens = []
    
    print("Precomputing text embeddings...")
    for text_files_batch in tqdm(all_text_files_batched, desc="Processing text files"):
        text_data = []
        for name in text_files_batch:
            text_data.extend(load_text_data(text_dir, name))

        if not text_data:
            continue
        
        with torch.no_grad():
            # 获取embedding和mask
            tensor, mask = text_model(text_data)  # text_model expects a list
            nb_tokens = mask.sum(dim=1)
            if torch.isnan(tensor).any():
                raise ValueError("contain NaN!!")
        
        t_lens += list(nb_tokens)
        all_texts = all_texts + text_data

        tensor_no_padding = [x[:n].cpu() for x, n in zip(tensor , nb_tokens)]
        tensor_concat = torch.cat(tensor_no_padding)

        big_tensor.append(tensor_concat)

        # save where it is
        ends = torch.cumsum(nb_tokens, 0)
        begins = torch.cat((0 * ends[[0]], ends[:-1]))

        # offset
        ends += nb_tokens_so_far
        begins += nb_tokens_so_far
        nb_tokens_so_far += len(tensor_concat)

        index.append(torch.stack((begins, ends)).T)

    big_tensor = torch.cat(big_tensor).cpu().numpy()
    index = torch.cat(index).cpu().numpy()
            
    # 保存预计算的数据
    text_model_name = text_model_name.replace('/', '_')
    ptpath = os.path.join(output_dir, f'{text_model_name}_{max_length}.npy')
    slicepath = os.path.join(output_dir, f'{text_model_name}_{max_length}_slice.npy')
    jsonpath = os.path.join(output_dir, f'{text_model_name}_{max_length}_index.json')

    np.save(ptpath, big_tensor)
    np.save(slicepath, index)
    
    print(f"{ptpath} written")
    print(f"{slicepath} written")

    # correspondance
    dico = {txt: i for i, txt in enumerate(all_texts)}
    with open(jsonpath, "w") as ff:
        ff.write(json.dumps(dico, indent=4))
    print(f"{jsonpath} written")

    print('mean', sum(t_lens)/len(t_lens))
    print('min', min(t_lens))
    print('max', max(t_lens))
    print(sum([1 for t in t_lens if t >77]))
    print(sum([1 for t in t_lens if t >64]))
    print(sum([1 for t in t_lens if t >32]))
    print(sum([1 for t in t_lens if t >20]))


def main():
    parser = argparse.ArgumentParser(description='Precompute text embeddings for motion datasets')
    parser.add_argument('--datasets', nargs='+', default=['HumanML3D', 'KIT-ML', 'Motion-X', 'CMP'], 
                        help='Dataset names to process')
    parser.add_argument('--text_model', default='flan-t5-large', 
                        help='Text model name (flan-t5-base, flan-t5-large, ViT-B/32, ViT-L/14 etc.)')
    parser.add_argument('--max_length', type=int, default=77, 
                        help='Maximum text length')
    parser.add_argument('--dataset_root', default='./dataset', 
                        help='Root directory of datasets')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
    
    args = parser.parse_args()
    
    device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))
    print(f"Using Device: {device}")

    for dataset_name in args.datasets:
        dataset_path = os.path.join(args.dataset_root, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}, skipping...")
            continue
            
        # try:
        precompute_embeddings_for_dataset(
            dataset_path, 
            args.text_model, 
            args.max_length,
            device
        )
        # except Exception as e:
        #     print(f"Error processing {dataset_name}: {e}")
        #     continue

if __name__ == '__main__':
    main()