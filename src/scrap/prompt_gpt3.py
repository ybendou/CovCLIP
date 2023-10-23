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

while error:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        if args.debug or args.load_file=='':
            scientificNames = {'1140837':"Cyclamen"}
        else:
            with open(args.load_file, 'r') as f:
                scientificNames = json.load(f)

        scientificNames = {k:v for k,v in scientificNames.items() if f'{k}.json' not in os.listdir(args.save_folder)}
        print('Starting from', len(scientificNames), 'species')
        for node_id, scientificName in tqdm(list(scientificNames.items())[:]):
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Give me a physical and visual description of the specie {scientificName} with its distinctive features",
            temperature=0.6,
            max_tokens=args.max_tokens,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1
            )
            if args.save_folder:
                with open(os.path.join(args.save_folder, f'{node_id}.json'), 'w') as f:
                    #f.write(response['choices'][0]['text'])
                    json.dump(response, f)
        error = False
    except Exception as e:
        error = True
        print(e)
        print('Error, retrying in 5 seconds')
        time.sleep(5)