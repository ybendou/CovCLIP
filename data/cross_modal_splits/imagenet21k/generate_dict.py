import json
import os

# wordnet ids:
ids = open('imagenet21k_wordnet_ids.txt').read().splitlines()
names = open('imagenet21k_wordnet_lemmas.txt').read().splitlines()
assert len(ids) == len(names), 'ids and names should have the same length'
# create dictionary
d = {"imagenet21k": {"prefix": "", "classes":dict(zip(ids, names))}}
# save dictionary
json.dump(d, open('imagenet21k_class_names.json', 'w'), indent=4)


