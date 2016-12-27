
# coding: utf-8

# In[6]:

import gensim
import json

model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 


# In[32]:

def word_vec_dict(word_list):
    word_dict = {}
    
    for word in word_list: 
        if "_" in word: 
            word1, word2 = word.split("_")
            word_dict[word] =  ((model[word1] + model[word2]) / 2.0).tolist()
        else:
            word_dict[word] = model[word].tolist()
    
    with open('word_dict.txt', 'w') as outfile:
        json.dump(word_dict, outfile, sort_keys=True, indent=4)
        
    return word_dict


# In[33]:

word_vec_dict(['arcade_machine', 'armchair', 'awning', 'bag', 'balcony', 'ball', 'barrel', 'basket', 'beam', 'bed', 'bench', 'billiard_table', 'board', 'boat', 'book', 'bookcase', 'books', 'bottle', 'bottles', 'bowl', 'box', 'boxes', 'bucket', 'building', 'buildings', 'bulletin_board', 'bus', 'bushes', 'cabinet', 'cabinets', 'can', 'candle', 'car', 'cars', 'ceiling', 'ceiling_fan', 'ceiling_lamp', 'chair', 'chandelier', 'clock', 'clothes', 'coffee_table', 'column', 'counter', 'countertop', 'cup', 'curtain', 'cushion', 'deck_chair', 'desk', 'desk_lamp', 'dishwasher', 'door', 'door_frame', 'double_door', 'drawer', 'extractor_hood', 'faucet', 'fence', 'field', 'flag', 'floor', 'floor_lamp', 'flowers', 'fluorescent_tube', 'gate', 'glass', 'grass', 'grille', 'ground', 'handrail', 'hat', 'hedge', 'hill', 'house', 'jar', 'keyboard', 'land', 'magazine', 'magazines', 'microwave', 'mirror', 'mountain', 'mug', 'napkin', 'night_table', 'ottoman', 'outlet', 'oven', 'painting', 'palm_tree', 'pane', 'paper', 'path', 'people', 'person', 'person_sitting', 'person_standing', 'person_walking', 'picture', 'pillow', 'pipe', 'plant', 'plant_pot', 'plants', 'plate', 'pole', 'poster', 'pot', 'purse', 'railing', 'refrigerator', 'river_water', 'road', 'rock', 'rocks', 'rocky_mountain', 'rug', 'sand_beach', 'sconce', 'screen', 'sculpture', 'sea_water', 'seat', 'seats', 'shelf', 'shelves', 'shoe', 'shoes', 'shop_window', 'showcase', 'shutter', 'side_table', 'sidewalk', 'sign', 'sink', 'sky', 'skyscraper', 'sneaker', 'snowy_mountain', 'sofa', 'spotlight', 'staircase', 'stand', 'statue', 'step', 'steps', 'stone', 'stones', 'stool', 'stove', 'streetlight', 'switch', 'swivel_chair', 'table', 'telephone', 'television', 'text', 'toilet', 'towel', 'toy', 'tray', 'tree', 'tree_trunk', 'trees', 'truck', 'umbrella', 'van', 'vase', 'wall', 'wardrobe', 'washbasin', 'water', 'window', 'worktop'])


# In[36]:

word_dict = word_vec_dict(['arcade_machine'])
len(word_dict['arcade_machine'])


# In[ ]:



