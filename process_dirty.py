import json

with open('./development_kit/data/object_categories.txt') as file:
    table = {}
    for line in file.readlines():
        word, index = line.split(' ')
        table[int(index)] = word


with open('object_index_to_string_mapping.txt', 'w') as file:
    json.dump(table, file, sort_keys=True, indent=4)


