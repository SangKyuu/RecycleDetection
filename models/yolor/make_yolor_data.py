import json
import tqdm
import cv2
import glob

def read_json(path):
  with open(path, 'r') as f:
    data = json.load(f)
    return data

def coco_pair_ImgAnn(coco_json):
  parent_dir = '/home/RecycleDetection/dataset/'

  img_dicts = [[] for i in range(len(coco_json['images']))]

  for ann in tqdm.tqdm(coco_json['annotations']):
    if len(img_dicts[ann['image_id']]) == 0:
      img_dicts[ann['image_id']].append(parent_dir + coco_json['images'][ann['image_id']]['file_name'])
      img_dicts[ann['image_id']].append([ann['category_id'],*ann['bbox']])
    else:
      img_dicts[ann['image_id']].append([ann['category_id'],*ann['bbox']])
  
  for img in tqdm.tqdm(img_dicts):
    h, w, _ = cv2.imread(img[0]).shape
    ann_list = [' '.join(map(str,[i[0], (i[1]+i[3]*0.5)/w, (i[2]+i[4]*0.5)/h, i[3]/w, i[4]/h]))+'\n' for i in img[1:]]
    with open(img[0].replace('jpg','txt'), 'w') as f:
      f.writelines(ann_list)


train_json = read_json('/home/RecycleDetection/dataset/train.json')
coco_pair_ImgAnn(train_json)


all_data = ['train', 'test']
parent_dir = '/home/RecycleDetection/dataset/'

for i in all_data:
  files = glob.glob(parent_dir+i+'/*.jpg')
  files = [p+'\n' for p in files]
  with open(parent_dir+i+'.txt', 'w') as f:
    f.writelines(files)
