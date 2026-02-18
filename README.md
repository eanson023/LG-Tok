# Language-Guided Transformer Tokenizer for Human Motion Generation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.08337-b31b1b.svg)](https://www.arxiv.org/abs/2602.08337)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://eanson023.github.io/LG-Tok)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)

</div>

This repository contains the official implementation of **LG-Tok**, a language-guided tokenizer for efficient text-driven human motion generation.

## 📰 News

- **[2026-02]** 🎉 Training code, inference code, and pretrained model weights are now fully released!

## 📝 TODO

- [ ] Release motion editing code

## Preparation

### 🔧 Installation

First, install `ffmpeg` for stick figure visualization:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

Next, create a conda environment with Python 3.10:

```bash
conda create -y -n lg-tok python=3.10
```

After the environment is created, activate it and install the required dependencies:

```bash
conda activate lg-tok
pip install -r requirements.txt
```

Note: The requirements.txt specifies PyTorch version 2.2.0.

Finally, install spacy dependencies for part-of-speech tagging:

```bash
python -m spacy download en_core_web_sm
```

### 🤖 Language Model

LG-Tok achieves efficient generation through language-guided tokenization. We use **LLaMA-3.2-1B** as the language encoder during the tokenization stage. 

Please visit https://huggingface.co/meta-llama/Llama-3.2-1B to fill out the form and request model access. After receiving approval from Meta, run:

```bash
mkdir deps
cd deps
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
cd ..
```

Alternatively, you can use `huggingface-cli` for downloading, which may be faster.

### 📊 Evaluation Models

We validated LG-Tok's effectiveness on two large-scale datasets: **HumanML3D** and **Motion-X**. Download the evaluation models using the following commands:

```bash
rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m
mkdir motionx

cd t2m 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1ejiz4NvyuoTj3BIdfNrTFFZBZ-zq4oKD/view?usp=sharing
echo -e "Unzipping humanml3d evaluators"
unzip evaluators_humanml3d.zip
echo -e "Cleaning humanml3d evaluators zip"
rm evaluators_humanml3d.zip

cd ../motionx 
echo -e "Downloading evaluation models for MotionX dataset"
gdown --fuzzy https://drive.google.com/file/d/1cazdW_r9Ma6XwGE3EgMjuo7BWLOHmPWJ/view?usp=sharing
echo -e "Unzipping motionx evaluators"
unzip evaluators_motionx.zip
echo -e "Cleaning motionx evaluators zip"
rm evaluators_motionx.zip

cd ../../
```

### 🎯 Download Pretrained Models

Download the pretrained checkpoints for both datasets using the following commands:

```bash
cd checkpoints/t2m
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/17x5pfdyJ9xuNnNDNJjUwFkX213Qsn5fl/view?usp=sharing
echo -e "Unzipping HumanML3D pretrained models (LG-Tok and the SAR Model)"
unzip pretrained_models_t2m.zip

cd ../motionx
echo -e "Downloading pretrained models for Motion-X dataset"
gdown --fuzzy https://drive.google.com/file/d/1JnoSLRuepnWOqA8q6dG1ZoVUs3QuH1SE/view?usp=sharing
echo -e "Unzipping Motion-X pretrained models (LG-Tok and the SAR Model)"
unzip pretrained_models_motionx.zip

rm pretrained_models_t2m.zip
rm pretrained_models_motionx.zip

cd ../../
```

### 📁 Datasets

#### HumanML3D

Please visit https://github.com/EricGuo5513/HumanML3D and follow the instructions to prepare the dataset. Finally, create a symbolic link to the `./dataset` folder.

#### Motion-X

Please visit https://github.com/IDEA-Research/Motion-X and follow the instructions to prepare the dataset. Then, convert the data representation following the instructions at https://github.com/IDEA-Research/HumanTOMATO/tree/main/src/tomato_represenation. Finally, place it in the `./dataset` folder.

*A simpler acquisition method will be provided in the future.*

Since we adopt a more compact representation, please run the following command to calculate new mean and standard deviation:

```bash
python utils/cal_mean_std.py
```

## 🎬 Demo

We provide an implementation for generating human motion. You can perform inference using our provided weights.

### HumanML3D:

```bash
python demo.py --name t2m_pkeep_rope_ffsize768_bs64_milestone100_200_trans_tok_v2_49tokens_nopadmask_encall \
  --text_prompt 'The woman walks on a balance beam.' \
  --repeat_times 10 \
  --motion_length 196 \
  --gpu_id 0 \
  --dataset_name t2m \
  --tfg 2.0
```

### Motion-X:

```bash
python demo.py --name t2m_pkeep_rope_ffsize768_bs64_milestone100_200_trans_tok_49tokens_nopadmask_encall \
  --text_prompt 'The woman walks on a balance beam.' \
  --repeat_times 10 \
  --motion_length 196 \
  --gpu_id 0 \
  --dataset_name motionx \
  --cond_scale 2 \
  --tfg 1
```

### Key Parameters:

- `cond_scale`: Classifier-Free Guidance (CFG) in the generative model. Optimal values: 4.0 for HumanML3D, 2.0 for Motion-X.
- `tfg`: Language-Free Decoding in the detokenizer (hyperparameter for the language-drop scheme mentioned in Section 3.4 of the paper). Optimal values: 2.0 for HumanML3D, 1.0 for Motion-X.

You can also adjust `--top_p`, `--top_k`, and `--temperature` parameters for sampling.

After execution, check the `./generation` folder for visualization results.

### 🎨 Visualization

`assets/LG-Tok.blend` is the template we used for motion visualization in our paper. You can follow the tutorial at https://github.com/EricGuo5513/momask-codes#dancers-visualization to complete the visualization.

Specifically, import the `.bvh` files from `./generation/text2motion` into the `.blend` project and follow the tutorial to complete the visualization.

## 📈 Evaluation

We currently provide anonymous weights for LG-Tok. LG-Tok-mini and LG-Tok-mid versions will be provided in the future.

### 🔄 Reconstruction

Run the following commands for evaluation:

**HumanML3D:**

```bash
python eval_tok.py --name trans_tok_mosa_Llama-3.2-1B_rope1d_base100_enc_ctx_ctx_dec_crs_crs_vit_patch1_llama_mtfg_ulen8_coin_txt_maeattnmask_correctinitweight \
  --dataset t2m \
  --gpu_id 0 \
  --which_epoch fid
```

**Motion-X:**

```bash
python eval_tok.py --name trans_tok_mosa_Llama-3.2-1B_rope1d_base100_enc_ctx_ctx_dec_crs_crs_vit_patch1_llama_mtfg_ulen8_coin_txt_maeattnmask_correctinitweight_womae \
  --dataset motionx \
  --gpu_id 0 \
  --which_epoch fid
```

### 🎯 Generation

LG-Tok aligns natural language with motion at the tokenization stage, producing compact semantic representations. With our Transformer-based architecture and language-drop scheme, LG-Tok achieves superior performance while using significantly fewer tokens.

<div align="center">
<img src="assets/table1.png" alt="Generation Results" width="90%">
</div>

Run the following commands for evaluation:

**HumanML3D:**

```bash
python eval_t2m.py --name t2m_pkeep_rope_ffsize768_bs64_milestone100_200_trans_tok_v2_49tokens_nopadmask_encall \
  --dataset t2m \
  --which_epoch fid \
  --tfg 2.0 \
  --gpu_id 0
```

**Motion-X:**

```bash
python eval_t2m.py --name t2m_pkeep_rope_ffsize768_bs64_milestone100_200_trans_tok_49tokens_nopadmask_encall \
  --dataset motionx \
  --which_epoch fid \
  --tfg 1.0 \
  --gpu_id 0
```

## 🚀 Training Your Own Model

### 💾 Pre-compute Text Embeddings [Optional]

To accelerate training, we provide a text embedding caching mechanism. You can run:

```bash
python prepare/prepare_text_embeddings.py \
  --datasets HumanML3D Motion-X \
  --text_model Llama-3.2-1B \
  --gpu_id 0
```

### 🎓 Training LG-Tok

**HumanML3D:**

```bash
python train_tok.py --name lg-tok \
  --gpu_id 0 \
  --batch_size 128 \
  --dataset_name t2m \
  --using_znorm \
  --eval_every_i 2000 \
  --unit_length 8 \
  --text_model Llama-3.2-1B \
  --text_max_len 77
```

**Motion-X:**

```bash
python train_tok.py --name lg-tok \
  --gpu_id 0 \
  --batch_size 128 \
  --dataset_name motionx \
  --using_znorm \
  --eval_every_i 2000 \
  --unit_length 8 \
  --text_model Llama-3.2-1B \
  --text_max_len 77
```

### 🏗️ Training Generative Model

**HumanML3D:**

```bash
python train_t2m.py --vq_name lg-tok \
  --dataset_name t2m \
  --name gen-model \
  --gpu_id 0 \
  --batch_size 64 \
  --ff_size 768 \
  --milestones 100 200
```

**Motion-X:**

```bash
python train_t2m.py --vq_name lg-tok \
  --dataset_name motionx \
  --name gen-model \
  --gpu_id 0 \
  --batch_size 64 \
  --ff_size 768 \
  --milestones 100 200
```

## 🙏 Acknowledgments

We stand on the shoulders of giants. Thanks to the following open-source repositories:

- [momask](https://github.com/EricGuo5513/momask-codes)
- [MARDM](https://github.com/neu-vi/MARDM)
- [2D-Positional-Encoding-Vision-Transformer](https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer)
- [Mesh-VQ-VAE](https://github.com/g-fiche/Mesh-VQ-VAE)
- [1d-tokenizer](https://github.com/bytedance/1d-tokenizer)
- And many more...

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yan2026language,
  title={Language-Guided Transformer Tokenizer for Human Motion Generation},
  author={Yan, Sheng and Wang, Yong and Du, Xin and Yuan, Junsong and Liu, Mengyuan},
  journal={arXiv preprint arXiv:2602.08337},
  year={2026}
}
```

