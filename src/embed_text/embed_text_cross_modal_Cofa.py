##################################################
"""
This file embeds the text obtained from Cofa which are multiple prompts from gpt3 for each class. 
Link of Cofa: https://github.com/ZrrSkywalker/CaFo
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
        elt = self.data[idx]
        chunk_size = len(elt) # start with the whole text
        if args.backbone == 'bert':
            elt = [self.tokenizer(t) for t in elt]
            if len(elt)>1:
                elt = [pad(t, chunk_size=chunk_size) for t in elt]
            elt = torch.stack(elt)
        else:
            elt = torch.stack([self.tokenizer(t) for t in elt])
        return elt, self.target_transforms(self.targets[idx]), chunk_size
    def __len__(self):
        return self.length

def clean(x):
    return x.replace('\n', '')

def pad(x, chunk_size=512):
    """
    pad tokens to allow for multiple batches
    """
    return F.pad(x, (0, max(0, chunk_size-x.shape[0])))
transform_dataset_name = {
"caltech_prompt_chat.json": "semanticfs_caltech_101_chat",  
"caltech_prompt.json": "semanticfs_caltech_101",  
"dtd_prompt.json": "semanticfs_dtd",  
"eurosat_prompt.json": "semanticfs_eurosat",  
"fgvc_prompt.json": "semanticfs_fgvc_aircraft_2013b",  
"food101_prompt.json": "semanticfs_food_101",  
"imagenet_prompt.json": "semanticfs_imagenet",  
"oxford_flowers_prompt.json": "semanticfs_oxford_flowers",  
"oxford_pets_prompt.json": "semanticfs_oxford_pets",  
"stanford_cars_prompt.json": "semanticfs_stanford_cars",  
"sun397_prompt.json": "semanticfs_sun397",  
"ucf101_prompt.json": "semanticfs_ucf101",
}
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
    
    # data is stored in multiple json files (one per class)
    # we load all the json files and store them in a dictionary
    # the keys are the class names and the values are the json files
    data_dict = {}
    files = os.listdir(args.text_path)
    for f in files:
        if f.endswith('.json'):
            single_file = json.load(open(os.path.join(args.text_path, f), 'r'))
            print(single_file)
            for i, (k, v) in enumerate(single_file.items()):
                data_dict[f'{transform_dataset_name[f]}_{i}'] = v

    dataset = DataHolder(data_dict, tokenizer=tokenizer, opener=None, chunk_size=512)
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
                features.append({"features":feats, "name_class":target[0].replace('.txt', '').replace('.json', '')}) # Average the chunks

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
        torch.save(features, f'{args.save_features}_with_tokens{args.return_tokens}_{args.backbone}_{name.replace("-", "").replace("/", "")}.pt')
##### remove n Chinese, it is called 银耳 (pinyin: yín ěr; literally "silver ear"), 雪耳 (pinyin: xuě ěr; literally "snow ear"); or 白木耳 (pinyin: bái mù ěr, l is too long for context length 77 """"" index is 200 