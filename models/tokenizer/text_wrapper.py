import os
import torch
import torch.nn as nn
import numpy as np
import orjson
import clip
from utils.tools import lengths_to_mask
from torch.nn.utils.rnn import pad_sequence


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEXT_MODEL_ZOO = {
    "clip": "ViT-B_32",
    "flan-t5-large" : "flan-t5-large",
    "ModernBERT-base" : "ModernBERT-base",
    "ModernBERT-large" : "ModernBERT-large",
    "clipl": "ViT-L_14",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "t5-large": "t5-large",
    "Llama-3.2-1B": "Llama-3.2-1B",
    "Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
}


def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class CLIPModelWrapper(nn.Module):
    def __init__(self, clip_version):
        super(CLIPModelWrapper, self).__init__()
        self.clip_model, _ = self.load_and_freeze_clip(clip_version)
        self.embed_dim = self.clip_model.text_projection.shape[0]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        print("***clip model loaded***")
        return clip_model, clip_preprocess

    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)

        # self.clip_model.encode_text
        x = self.clip_model.token_embedding(text).type(
            self.clip_model.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # sent_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        # sentence level and word level
        # max_len = max(word_mask.sum(dim=-1))
        # x = x[:, :max_len]
        # word_mask = word_mask[:, :max_len]
        # return sent_x.float() # x.float()
        text_mask = torch.ones((x.shape[:2])).to(x.device).bool()
        return x.float(), text_mask


class T5ModelWrapper(nn.Module):
    def __init__(self, t5_version, max_length=120):
        super(T5ModelWrapper, self).__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        from_pretrained = os.path.join('./deps', t5_version)
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            local_files_only = True,
            legacy=False
        )
        
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained,
            local_files_only=True,
        ).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_length = max_length
        self.embed_dim = self.model.config.d_model  # 768 for T5-base, 1024 for T5-large, etc.


    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        # enc = tokenizer(text_caption, return_tensors="pt")
        enc = self.tokenizer(raw_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).to(device)
        attention_mask = enc["attention_mask"].to(device).bool()

        # forward pass through encoder only
        with torch.no_grad():
            encoded = self.model(**enc).last_hidden_state.detach()  # (B, Nt, D)

        return encoded, attention_mask
    

class BertModelWrapper(nn.Module):
    def __init__(self, bert_version, max_length=120):
        super(BertModelWrapper, self).__init__()
        from transformers import AutoTokenizer, AutoModel

        from_pretrained = os.path.join('./deps', bert_version)
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            local_files_only = True,
            legacy=False
        )
        
        self.model = AutoModel.from_pretrained(
            from_pretrained,
            local_files_only=True
        ).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_length = max_length
        self.embed_dim = self.model.config.hidden_size 


    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        # enc = tokenizer(text_caption, return_tensors="pt")
        enc = self.tokenizer(raw_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).to(device)
        attention_mask = enc["attention_mask"].to(device).bool()

        # forward pass through encoder only
        with torch.no_grad():
            encoded = self.model(**enc).last_hidden_state.detach()  # (B, Nt, D)

        return encoded, attention_mask


class ModernBertModelWrapper(nn.Module):
    def __init__(self, bert_version, max_length=120):
        super(ModernBertModelWrapper, self).__init__()

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)  # Force math implementation

        from transformers import AutoTokenizer, ModernBertModel

        from_pretrained = os.path.join('./deps', bert_version)
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            local_files_only = True,
            legacy=False
        )
        
        self.model = ModernBertModel.from_pretrained(
            from_pretrained,
            local_files_only=True
        ).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_length = max_length
        self.embed_dim = self.model.config.hidden_size 


    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        # enc = tokenizer(text_caption, return_tensors="pt")
        enc = self.tokenizer(raw_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).to(device)
        attention_mask = enc["attention_mask"].to(device).bool()

        # forward pass through encoder only
        with torch.no_grad():
            encoded = self.model(**enc).last_hidden_state.detach()  # (B, Nt, D)

        return encoded, attention_mask


class LlamaModelWrapper(nn.Module):
    def __init__(self, llama_version, max_length=120):
        super(LlamaModelWrapper, self).__init__()
        from transformers import AutoTokenizer, LlamaModel

        from_pretrained = os.path.join('./deps', llama_version)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            local_files_only=True,  
            legacy=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LlamaModel.from_pretrained(
            from_pretrained,
            local_files_only=True, 
            # torch_dtype=torch.float16, 
            # device_map="auto" 
        ).eval()
        
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_length = max_length
        self.embed_dim = self.model.config.hidden_size 

    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        
        enc = self.tokenizer(
            raw_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        ).to(device)
        
        attention_mask = enc["attention_mask"].to(device).bool()

        with torch.no_grad():
            outputs = self.model(**enc)
            encoded = outputs.last_hidden_state.detach()  # (B, Nt, D)

        return encoded, attention_mask




class QwenModelWrapper(nn.Module):
    def __init__(self, llama_version, max_length=120):
        super(QwenModelWrapper, self).__init__()
        from transformers import AutoTokenizer, AutoModel

        from_pretrained = os.path.join('./deps', llama_version)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            local_files_only=True,  
            legacy=False,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(
            from_pretrained,
            local_files_only=True, 
            # torch_dtype=torch.float16, 
            # device_map="auto" 
        ).eval()
        
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_length = max_length
        self.embed_dim = self.model.config.hidden_size 

    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        
        enc = self.tokenizer(
            raw_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        ).to(device)
        
        attention_mask = enc["attention_mask"].to(device).bool()

        with torch.no_grad():
            outputs = self.model(**enc)
            encoded = outputs.last_hidden_state.detach()  # (B, Nt, D)

        return encoded, attention_mask



class TextEmbedding(nn.Module):
    def __init__(self, args, text_model_name, max_length=120):
        super(TextEmbedding, self).__init__()

        self.text_model_name = text_model_name
        self.load_text_embeddings(args.text_emb_dir, text_model_name, max_length)
        self.max_length = max_length
        self.device = args.device

        if not self.pre_text_emb:
            self.load_model(text_model_name, max_length)
            print(f"loading {text_model_name} from scratch")
            
    
    def load_model(self, text_model, max_length):
        if 'clip' in text_model:
            self.text_model = CLIPModelWrapper(text_model)
            self.max_length = 77
        elif 't5' in text_model:
            self.text_model = T5ModelWrapper(text_model, max_length)
        elif 'bert' in text_model:
            self.text_model = BertModelWrapper(text_model, max_length)
        elif 'Llama' in text_model:
            self.text_model = LlamaModelWrapper(text_model, max_length)
        else:
            raise NotImplementedError(f"Unsupported text model: {text_model}")
        
        self.embed_dim = self.text_model.embed_dim
    

    def load_text_embeddings(self, text_emb_dir, text_model, max_length):
        try:
            emb_path = os.path.join(text_emb_dir, TEXT_MODEL_ZOO[text_model] + f"_{max_length}.npy")
            self.text_emb_big =  torch.from_numpy(np.load(emb_path)).to(dtype=torch.float, device='cpu')
            if torch.isnan(self.text_emb_big).any():
                raise ValueError("contain NaN!!")
            self.text_emb_slice = np.load(
                os.path.join(text_emb_dir, TEXT_MODEL_ZOO[text_model] + f"_{max_length}_slice.npy")
            )
            self.text_emb_index = load_json(
                os.path.join(text_emb_dir, TEXT_MODEL_ZOO[text_model] + f"_{max_length}_index.json")
            )
            self.embed_dim = self.text_emb_big.shape[-1]
            self.pre_text_emb = True
            print(f"Preloading text embeddings loaded from {emb_path}")
        except Exception as e:
            print(f"Error loading text embeddings: {e}")
            self.pre_text_emb = False


    def get_embedding(self, texts):
        try:
            # Precomputed in advance
            embeddings = []
            lens = []
            for text in texts:
                index = self.text_emb_index[text]
                begin, end = self.text_emb_slice[index]
                embedding = self.text_emb_big[begin:end]
                embeddings.append(embedding)
                lens.append(len(embedding))
            
            embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
            embeddings = torch.cat([embeddings,
                                            torch.zeros((embeddings.shape[0], self.max_length - embeddings.shape[1], self.embed_dim))
                                        ], dim=1).to(self.device)

            attention_mask = lengths_to_mask(torch.tensor(lens, device=self.device), self.max_length)
        except Exception as e:
            print(f"Error retrieving precomputed embeddings: {e}")
            return None, None

        return embeddings, attention_mask

    def forward(self, raw_text):
        if self.pre_text_emb:

            result = self.get_embedding(raw_text)
            if None not in result:
                return result
            # Text not found in precomputed embeddings, need to load model
            print(f"Text not found in precomputed embeddings, loading text model on-demand...")
            if not hasattr(self, 'text_model'):
                self.load_model(self.text_model_name, self.max_length)
                self.text_model.to(self.device)
        return self.text_model(raw_text)
