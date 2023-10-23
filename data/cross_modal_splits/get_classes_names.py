import json
import os
map_splits_to_datasets = {
    'split_zhou_Caltech101.json':['caltech-101', ''],
    'split_zhou_DescribableTextures.json': ['dtd', 'texture'],
    'split_zhou_EuroSAT.json': ['eurosat', 'satelite image'],
    'split_zhou_Food101.json':['food-101', 'food'],
    'split_zhou_OxfordFlowers.json': ['oxford_flowers', 'flower'], # all images are in the same folder 
    'split_zhou_OxfordPets.json': ['oxford_pets', ''], # all images are in the same folder
    'split_zhou_StanfordCars.json': ['stanford_cars', 'car'],
    'split_zhou_SUN397.json': ['sun397', ''],
    'split_zhou_UCF101.json': ['ucf101', '']
}
def clean_text(text):
    return text.replace('_', ' ')
def get_class_names(split):
    split = list(map(lambda x: f'{x[1]}</>{x[2]}', split)) # class_id</>class_name
    split = list(set(split)) # remove duplicates
    split = list(map(lambda x: x.split('</>'), split)) # split class_id and class_name
    split = list(map(lambda x: (int(x[0]), clean_text(x[1])), split)) # convert class_id to int and clean text 
    return dict(split)

# json files path
files = os.listdir('.')
class_names = {}
for file in files:
    if file.endswith('.json') and 'split_zhou' in file:
        name = f"semanticfs_{map_splits_to_datasets[file][0].replace('-', '_').replace(' ', '_')}"
        dic = json.load(open(file, 'r'))
        for dataset,split in list(dic.items())[0:1]:
            class_names[f'{name}'] = {"prefix": map_splits_to_datasets[file][1], "classes":get_class_names(split)}

# add aircraft and imagenet
class_names["semanticfs_imagenet"] = json.load(open('class_names_imagenet.json', 'r'))
class_names["semanticfs_imagenet"]["classes"] = {i:v for i,v in enumerate(class_names["semanticfs_imagenet"]["classes"])}
class_names["semanticfs_fgvc_aircraft_2013b"] = json.load(open('class_names_aircraft.json', 'r'))
class_names["semanticfs_fgvc_aircraft_2013b"]["classes"] = {i:v for i,v in enumerate(sorted(class_names["semanticfs_fgvc_aircraft_2013b"]["classes"]))}

json.dump(class_names, open('class_names.json', 'w'), indent=4)