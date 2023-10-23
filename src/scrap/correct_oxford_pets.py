import json 
import os
import sys
import argparse
MYSPACE = os.environ['MYSPACE']
parser = argparse.ArgumentParser()
parser.add_argument("--json-path", type=str, default=os.path.join(MYSPACE, "few-shot-inaturalist-hf/text/gpt3_prompts_cross_modal/"))
args = parser.parse_args("")

class_names = json.load(open('../../data/cross_modal_splits/class_names.json', 'r'))['semanticfs_oxford_pets']['classes']
files = os.listdir(args.json_path)
for file in files:
    if 'oxford_pets' in file:
        # open file
        with open(os.path.join(args.json_path, file), 'r') as f:
            data = json.load(f)
        id = file.split('_')[-1].replace('.json', '')
        name = class_names[id]
        text = data['choices'][0]['text']
        print(f'A photo of a {name}{text}')
        # overwrite file
        data['choices'][0]['text'] = f'A photo of a {name}{text}'
        with open(os.path.join(args.json_path, file), 'w') as f:
            json.dump(data, f)
            