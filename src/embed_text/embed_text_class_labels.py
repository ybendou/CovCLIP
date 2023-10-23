##################################################
"""
python -i embed_text.py --text-path $MYSPACE/few-shot-inaturalist-hf/text/gpt3prompts/  --kingdom all --text-is-gpt3 --save-features ""
"""
##################################################
import torch
import argparse
import torch.nn as nn
import os 
from torchvision import transforms
import torch.nn.functional as F
import json
from tqdm import tqdm
from functools import partial
from PIL import Image
class Clip_Text(nn.Module):
    def __init__(self, name, device, return_tokens=False):
        super(Clip_Text, self).__init__()
        self.backbone = clip.load(name, device=device)[0]
        self.return_tokens=return_tokens
    def forward(self, text):
        cls_token = self.backbone.encode_text(text)
        if self.return_tokens:
            x = self.backbone.token_embedding(text).type(self.backbone.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.backbone.positional_embedding.type(self.backbone.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.backbone.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.backbone.ln_final(x).type(self.backbone.dtype)
            out = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        else:  
            out = cls_token
        return out

class DataHolder():
    def __init__(self, data, tokenizer, target_transforms=lambda x:x, opener=lambda x: Image.open(x).convert('RGB'), chunk_size=512):
        self.data = list(data.values())
        self.targets = list(data.keys())
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.opener = opener
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        elt = f'{args.prefix}{self.data[idx]}.'
        chunk_size = min(self.chunk_size, len(elt.split())) # start with the whole text
        # try to create chunks of size chunk_size, if it fails, reduce chunk_size by 10 until it works.
        chunked_elt = create_chunks(elt, chunk_size)
        if args.backbone == 'bert':
            chunked_elt = [self.tokenizer(t) for t in chunked_elt]
            if len(chunked_elt)>1:
                chunked_elt = [pad(t, chunk_size=chunk_size) for t in chunked_elt]
            chunked_elt = torch.stack(chunked_elt)
        else:
            chunked_elt = torch.stack([self.tokenizer(t) for t in chunked_elt])
        return chunked_elt, self.target_transforms(self.targets[idx]), chunk_size
    def __len__(self):
        return self.length

def create_chunks(text, chunk_size=512):
    """
    Split a text into multiple chunks if size exceed the chunk size limit.
    Returns a list containing multiple text chunks with max size being chunk_size.
    """
    initial_chunks = text.split('.')
    initial_chunks = [c[1:] if c[0]==' ' else c for c in initial_chunks if c!=''] # clean text if starts with a space.
    chunks = []
    chunk = ''

    while len(initial_chunks)>0:
        text_candidate = initial_chunks.pop(0)
        if len(text_candidate.split()) > chunk_size: 
            initial_chunks = [' '.join(text_candidate.split()[:chunk_size]), ' '.join(text_candidate.split()[chunk_size:])]+initial_chunks
            continue # if the text is too long, split it in two and add it to the initial chunks
        chunk_candidate = chunk+'.'*(chunk!='')+text_candidate
        merging = len(chunk_candidate.split())<=chunk_size
        if merging:
            chunk = chunk_candidate
        else: # if adding the new text exceeds max size, start a new chunk
            initial_chunks = [text_candidate]+initial_chunks
            chunks.append(chunk)
            chunk = ''
        if len(initial_chunks) == 0:
            if merging: # if there is still room where to add the last chunk and save the last chunk
                chunks.append(chunk)
            else:
                chunks.append(text_candidate) # otherwise create a new chunk
    return chunks

def clean(x):
    return x.replace('\n', '')

def pad(x, chunk_size=512):
    """
    pad tokens to allow for multiple batches
    """
    return F.pad(x, (0, max(0, chunk_size-x.shape[0])))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text-path', type=str, default='/home/y17bendo/campus/These/code/iNaturalist/clean_text_v2/')
    parser.add_argument('--save-features', type=str, default='')
    #parser.add_argument('--path-mapp', type=str, default='/home/y17bendo/campus/These/code/iNaturalist/map_class_names_to_urls_v2.json', help='path to the json file containing the mapping between class names and class identifiers')
    parser.add_argument('--backbone', type=str, default='clip')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--kingdom', type=str, default='all', help='kingdom to embed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--text-is-gpt3', action='store_true')
    parser.add_argument('--prefix', type=str, default='A photo of a ', help='prefix to add to the text (e.g. "A photo of a "')
    parser.add_argument('--return_tokens', action='store_true')

    args = parser.parse_args()
    if args.backbone == 'bert':
        from transformers import BertModel, BertTokenizer
        # Load the BERT model and the BERT tokenizer
        backbone = BertModel.from_pretrained('bert-base-uncased')
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = lambda x: torch.tensor(bert_tokenizer.encode(x))
        embed_dim = 768
        chunk_size = 512
    elif 'clip' in args.backbone:
        import clip
        tokenizer = clip.tokenize
        chunk_size = 200
        # if args.backbone =='clip' or 'b32' in args.backbone:
        #     name = 'ViT-B/32'
        #     embed_dim = 512
        # elif 'b16' in args.backbone:
        #     name = 'ViT-B/16'
        #     embed_dim = 512
        # elif 'l14' in args.backbone:
        #     name = 'ViT-L/14'
        #     embed_dim = 768
        # elif 'l14_336px' in args.backbone:
        #     name = 'ViT-L/14@336px'
        #     embed_dim = 768
        name, embed_dim = {"clip_rn50": ['RN50', 1024],
                            "clip_rn101": ['RN101', 512],
                            "clip_rn50x4": ['RN50x4', 640],
                            "clip_rn50x16": ['RN50x16', 768],
                            "clip_rn50x64": ['RN50x64', 1024],
                            "clip_b32": ['ViT-B/32', 512],
                            "clip": ['ViT-B/32', 512],
                            "clip_b16": ['ViT-B/16', 512],
                            "clip_l14": ['ViT-L/14', 768],
                            "clip_l16_336px": ['ViT-L/14@336px', 768]}[args.backbone.lower()]
        backbone = Clip_Text(name, args.device, return_tokens=args.return_tokens)
    MAX_BATCH_SIZE = 5000

    file = json.load(open(args.text_path, 'r'))    
    #mapp_name_to_identifier = json.load(open(args.path_mapp))
    #targets = [int(mapp_name_to_identifier[t.replace('.txt', '')]) for t in text_folder]
    dataset = DataHolder(file, tokenizer=tokenizer, opener=None, chunk_size=512)
    print(f'Loaded text file with: {len(file)} elements, dataset size: {len(dataset)}')
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = min(os.cpu_count(), 8))

    # Calculate features
    backbone.eval()
    backbone = backbone.to(args.device)

    iterator = enumerate(dataloader)
    features = []
    len_dataset = 5 if args.debug else len(dataloader)

    if 'clip' in args.backbone:
        for _ in tqdm(range(len_dataset)):
            batchIdx, (data, target, chunk_size) = next(iterator)
            data = data.to(args.device).squeeze(2).squeeze(0)
            if data.shape[0]>MAX_BATCH_SIZE:
                data = data[:MAX_BATCH_SIZE]
            with torch.no_grad():
                inner_batch_size = 224
                if data.shape[0]<=inner_batch_size:
                    feats = []
                    for i in range(0, data.shape[0], inner_batch_size):
                        feats.append(backbone(data[i:i+inner_batch_size]))
                    feats = torch.cat(feats, dim=0)
                else:
                    feats = backbone(data)
                features.append({"features":feats.reshape(-1, embed_dim), "name_class":target[0].replace('.txt', '').replace('.json', '')}) # Average the chunks

    elif args.backbone=='bert':
        for _ in tqdm(range(len_dataset)):
            batchIdx, (data, target, chunk_size) = next(iterator)
            data = data.to(args.device).reshape(-1, data.shape[-1])
            if data.shape[0]>MAX_BATCH_SIZE:
                data = data[:MAX_BATCH_SIZE]
            with torch.no_grad():
                feats_last_hidden_state = backbone(data).last_hidden_state
                feats_average = backbone(data).pooler_output
                features.append({"features average":feats_average.reshape(-1, embed_dim), "features last_hidden_state":feats_last_hidden_state, "name_class":target[0].replace('.txt', '').replace('.json', '')}) # Average the chunks

    # Save features
    if args.save_features != '':
        torch.save(features, f'{args.save_features}_{args.kingdom}_text_{args.backbone}.pt')
##### remove n Chinese, it is called 银耳 (pinyin: yín ěr; literally "silver ear"), 雪耳 (pinyin: xuě ěr; literally "snow ear"); or 白木耳 (pinyin: bái mù ěr, l is too long for context length 77 """"" index is 200 