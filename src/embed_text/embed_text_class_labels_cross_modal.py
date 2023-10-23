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
        chunked_elt = [elt]
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


def clean(x):
    return x.replace('\n', '')

def pad(x, chunk_size=512):
    """
    pad tokens to allow for multiple batches
    """
    return F.pad(x, (0, max(0, chunk_size-x.shape[0])))
vowels = ['a', 'e', 'i', 'o', 'u']
def ufc101(x):
    """
    Identify wether the class name is a verb or not, if it is, return 'someone' as prefix
    """
    splits = x.split(' ')
    is_verb = False
    if splits[0].lower() in ['apply', 'clean', 'haircut', 'tai', 'punch']:
        if splits[0].lower() == 'apply':
            output = f"someone applying {' '.join(splits[1:])}"
        else:
            output = {'clean': 'someone performing a Clean and Jerk in crossfit', 'haircut': 'someone giving a haircut', 'punch': 'a boxer giving a punch', 'tai': 'someone doing Tai and Chi'}[splits[0].lower()]
        return output
    for split in splits:
        if len(split)>=2:
            if split[-3:] == 'ing':
                is_verb = True
    if is_verb:
        return f'someone {x}'
    else:
        if x[0] in vowels:
            return f'someone doing an {x}'
        else:
            return f'someone doing a {x}'
        
queries = {
    "semanticfs_oxford_pets": lambda x: f"a photo of a {x}, a type of pet.",
    "semanticfs_sun397": lambda x: f"a photo of a {x}.",
    "semanticfs_eurosat": lambda x: f"a centered satellite photo of {x}.",
    "semanticfs_dtd": lambda x: f"{x} texture.",
    "semanticfs_oxford_flowers": lambda x: f"a photo of a {x}, a type of flower.",
    "semanticfs_caltech_101": lambda x: f"a photo of a {x}.",
    "semanticfs_food_101": lambda x: f"a photo of {x}, a type of food.",
    "semanticfs_stanford_cars": lambda x: f"a photo of a {x}.",
    "semanticfs_ucf101": lambda x: f"a photo of a person doing {x}.",
    "semanticfs_imagenet": lambda x: f"a photo of a {x}.",
    "semanticfs_fgvc_aircraft_2013b": lambda x: f"a photo of {x}, a type of aircraft.",
    "imagenet21k": lambda x: f"a photo of a {x.replace('_', ' ')}.",
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
        print(f'Backbone: {name} with {embed_dim} dimensions')
        backbone = Clip_Text(name, args.device, return_tokens=args.return_tokens)

    MAX_BATCH_SIZE = 5000
    names = json.load(open(args.text_path, 'r'))
    all_features = {}
    for dataset_name, dataset in names.items():
        keys, values = [], []
        for node_id, class_name in dataset['classes'].items():
            keys.append(class_name)
            values.append(queries[dataset_name](class_name))
        file = dict(zip(keys, values))
        #mapp_name_to_identifier = json.load(open(args.path_mapp))
        #targets = [int(mapp_name_to_identifier[t.replace('.txt', '')]) for t in text_folder]
        dataset = DataHolder(file, tokenizer=tokenizer, opener=None, chunk_size=512)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = min(os.cpu_count(), 8))

        # Calculate features
        backbone.eval()
        backbone = backbone.to(args.device)

        iterator = enumerate(dataloader)
        features = {}
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
                    features[target[0]] = feats.cpu() # Average the chunks
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
        all_features[dataset_name] = features
    # Save features
    if args.save_features != '':
        torch.save(all_features, f'{args.save_features}_with_tokens{args.return_tokens}.pt')
##### remove n Chinese, it is called 银耳 (pinyin: yín ěr; literally "silver ear"), 雪耳 (pinyin: xuě ěr; literally "snow ear"); or 白木耳 (pinyin: bái mù ěr, l is too long for context length 77 """"" index is 200 