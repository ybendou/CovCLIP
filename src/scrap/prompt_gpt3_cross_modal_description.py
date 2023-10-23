import os
import openai
import argparse
from tqdm import tqdm 
import json
import time
parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='')
parser.add_argument('--load-file', type=str, default='')
parser.add_argument("--debug", action="store_true", help="debugging mode")
parser.add_argument('--max-tokens', type=int, default=400)
args = parser.parse_args()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
error = True
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
    "semanticfs_oxford_pets": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} with its distinctive features",
    "semanticfs_sun397": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} with its distinctive features",
    "semanticfs_eurosat": lambda x: f"Give me a physical and visual description of a satelite image of {'an' if x[0] in vowels else 'a'} {x} with its distinctive features",
    "semanticfs_dtd": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} texture with its distinctive features?",
    "semanticfs_oxford_flowers": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} flower with its distinctive features?",
    "semanticfs_caltech_101": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} with its distinctive features",
    "semanticfs_food_101": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} with its distinctive features",
    "semanticfs_stanford_cars": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} car with its distinctive features",
    "semanticfs_ucf101": lambda x: f"Give me a physical and visual description of {ufc101(x)} with its distinctive features",
    "semanticfs_imagenet": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} with its distinctive features",
    "semanticfs_fgvc_aircraft_2013b": lambda x: f"Give me a physical and visual description of {'an' if x[0] in vowels else 'a'} {x} aircraft with its distinctive features",
}
while error:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        names = json.load(open(args.load_file)) 
        for dataset_name, dataset in names.items():
            print('dataset:', dataset_name)
            query = queries[dataset_name]           
            for node_id, class_name in tqdm(dataset['classes'].items()):
                save_file = f'{dataset_name}_{node_id}.json'
                prompt = query(class_name)
                if not os.path.exists(os.path.join(args.save_folder, save_file)) and not args.debug:
                    response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=query(class_name),
                    temperature=0.6,
                    max_tokens=args.max_tokens,
                    top_p=1,
                    frequency_penalty=1,
                    presence_penalty=1
                    )
                    if args.save_folder:
                        with open(os.path.join(args.save_folder, save_file), 'w') as f:
                            json.dump(response, f)
            error = False
    except Exception as e:
        error = True
        print(e)
        print('Error, retrying in 5 seconds')
        time.sleep(5)