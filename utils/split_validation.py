import json
import random

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def sort_by_image(annotations):
    img_ann_dict = {}
    for i in annotations:
        if i['image_id'] not in img_ann_dict:
            img_ann_dict[i['image_id']] = [i]
        else:
            img_ann_dict[i['image_id']].append(i)
    return img_ann_dict

def split_data(images_json, sorted_ann, n=5):
    random.seed(1000)
    random.shuffle(images_json)

    total_len = len(images_json)
    sub_len = total_len // n
    splitted_data = []
    for i in range(n):
        block = images_json[i*sub_len:(i+1)*sub_len] if i!=n-1 else images_json[i*sub_len:]
        ann_block = [sorted_ann[img_json['id']] for img_json in block]
        splitted_data.append([block, ann_block])

    return splitted_data

if __name__ == '__main__':
    n = 3
    train_json = load_json('../dataset/train.json')
    sorted_ann = sort_by_image(train_json['annotations'])
    n_data = split_data(train_json['images'], sorted_ann, n=n)

    for i, [img_json, ann_json] in enumerate(n_data):
        splitted_train = {'info':train_json['info'],
                          'licenses':train_json['licenses'],
                          'images':img_json,
                          'categories':train_json['categories'],
                          'annotations':ann_json}
        write_json(f'./dataset/train_block_({i+1}_{n}).json', splitted_train)
