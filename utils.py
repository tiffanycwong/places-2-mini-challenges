from skimage import io
import numpy as np
import random
from sklearn.utils import shuffle
import scipy.ndimage
import matplotlib.pyplot as plt
from memoize import *
from process_XML import extract_XML_obj
import json

@memoize
def load_word_embeddings():
    with open('./word_dict.txt') as file:
        word_dict = json.load(file)
    return word_dict

@memoize
def load_index_to_object_mapping():
    with open('object_index_to_string_mapping.txt') as file:
        mapping = json.load(file)
    return mapping

def get_data_for_batch(split_name, batch_index, batch_size, train_indices, load_small=False, load_easy=False, resize_dim=None):
    image_path_label_pairs = get_image_path_label_pairs(split_name, load_easy, load_small)
    objects = get_objects(split_name, load_easy, load_small)
    num_examples = len(image_path_label_pairs)
    start_index = batch_index*batch_size
    end_index = min(num_examples, (batch_index+1)*batch_size)
    X_batch = []
    scene_category = []
    object_encodings = []
    word_embeddings_averages = []
    Y_batch = {}
    word_embedding_dictionary = load_word_embeddings()
    index_to_object_mapping = load_index_to_object_mapping()

    for i in xrange(start_index, end_index):
        idx = train_indices[i]
        # idx = batch_indicies[i]
        path, label = image_path_label_pairs[idx]
        image = io.imread(path)
        image = image_process(image, resize_dim)
        X_batch.append(image)
        label_one_hot = one_hot_encoding(int(label), 100)
        scene_category.append(label_one_hot)
        objects_for_this = objects[idx]
        object_encoding = np.zeros(1000)
        word_embedding_average = np.zeros(300)
        for obj in objects_for_this:
            object_class, min_x, max_x, min_y, max_y = obj
            object_encoding[int(object_class)] += 1.0
            object_class_string = index_to_object_mapping[str(object_class)]
            word_embedding = np.array(word_embedding_dictionary[object_class_string])
            word_embedding_average += word_embedding

        word_embedding_average /= len(objects_for_this)

        object_encodings.append(object_encoding)
        word_embeddings_averages.append(word_embedding_average)


    Y_batch['scene_category'] = np.array(scene_category)
    Y_batch['object_encodings'] = np.array(object_encodings)
    Y_batch['word_embeddings_averages'] = np.array(word_embeddings_averages)

    X_batch = np.array(X_batch)

    return X_batch, Y_batch

@memoize
def get_objects(split_name, load_easy, load_small):
    easy_str = "_easy" if load_easy else ""
    scene_mapping_path = './development_kit/data/{}{}.txt'.format(split_name, easy_str)
    object_paths = ["./data/objects/"+row.split(' ')[0].strip().replace(".jpg",".xml") for row in open(scene_mapping_path).readlines()]
    if load_small:
        object_paths = object_paths[:500]
    objects = map(extract_XML_obj, object_paths)
    return objects

def image_process(image, resize_dim):
    processed = image
    if resize_dim:
        processed = scipy.misc.imresize(processed, size=(resize_dim, resize_dim))
    processed = processed.astype(float)/255.0 - 0.5
    return processed

def one_hot_encoding(label, num_categories):
    one_hot = np.zeros((num_categories))
    one_hot[label] = 1.0
    return one_hot

@memoize
def get_image_path_label_pairs(split_name, load_easy, load_small):
    print('reading {} data'.format(split_name))
    easy_str = "_easy" if load_easy else ""
    scene_mapping_path = './development_kit/data/{}{}.txt'.format(split_name, easy_str)
    image_path_label_pairs = [("./data/images/"+row.split(' ')[0].strip(),row.split(' ')[1].strip()) for row in open(scene_mapping_path).readlines()]
    if load_small:
        image_path_label_pairs = image_path_label_pairs[:500]
    return image_path_label_pairs

def get_scene_category_mappings():
    scene_mapping_path = './development_kit/data/categories.txt'
    scene_to_id_mapping = {row.split(' ')[0].strip():row.split(' ')[1].strip() for row in open(scene_mapping_path).readlines()}
    id_to_scene_mapping = {value:key for key, value in scene_to_id_mapping}
    return scene_to_id_mapping, id_to_scene_mapping

def get_batch(X, Y, batch_size, batch_index, augment_training_data):
    n_examples = len(X)
    start_index, end_index = get_start_end_index_for_batch(batch_size, batch_index, n_examples)
    X_batch = X[start_index:end_index]
    Y_batch = {key:Y[key][start_index:end_index] for key, value in Y.iteritems()}

    return X_batch, Y_batch

def get_start_end_index_for_batch(batch_size, batch_index, n_examples):
    start_index = batch_index*batch_size
    end_index = (batch_index+1)*batch_size
    batch_size_smart = min(batch_size, batch_size + (n_examples - end_index)) # if at end of data, use smaller batch size
    end_index_smart = batch_index*batch_size + batch_size_smart
    return start_index, end_index_smart

def shuffle_order(X, Y=None):
        shuffle_seed = random.randint(0, 1e5)
        index_shuf = range(len(X))
        shuffled = shuffle(index_shuf, random_state=shuffle_seed)
        X_shuffled = np.array([X[i] for i in index_shuf])
        if Y:
            Y_shuffled = {key:np.array([Y[key][i] for i in index_shuf]) for key in Y.keys()}
            return X_shuffled, Y_shuffled
        else:
            return X_shuffled


