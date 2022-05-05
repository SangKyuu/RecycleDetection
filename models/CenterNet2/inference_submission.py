import os
import copy
import torch
import cv2
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from centernet.config import add_centernet_config


# Register Dataset
try:
    register_coco_instances('coco_trash_test', {}, '../../dataset/test.json', '../../dataset/')
except AssertionError:
    pass

# config 불러오기
cfg = get_cfg()
add_centernet_config(cfg)
cfg.merge_from_file('./configs/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST.yaml')


# config 수정하기
cfg.DATASETS.TEST = ('coco_trash_test',)

cfg.DATALOADER.NUM_WOREKRS = 2

cfg.OUTPUT_DIR = './output'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'CenterNet2', 'CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST', 'model_final.pth')
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = cfg.MODEL.CENTERNET.INFERENCE_TH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.MODEL.CENTERNET.INFERENCE_TH
if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
    cfg.MODEL.CENTERNET.INFERENCE_TH = cfg.MODEL.CENTERNET.INFERENCE_TH
    cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = cfg.MODEL.CENTERNET.INFERENCE_TH

# model
predictor = DefaultPredictor(cfg)


# mapper - input data를 어떤 형식으로 return할지
def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    dataset_dict['image'] = image

    return dataset_dict


# test loader
test_loader = build_detection_test_loader(cfg, 'coco_trash_test', MyMapper)

with open('../../dataset/test.json', 'r') as f:
    test_json = json.load(f)

# output 뽑은 후 sumbmission 양식에 맞게 후처리
prediction_strings = []
file_names = []

class_num = 10
color_dict = {i: tuple(map(int,np.random.choice(range(256), size=3))) for i in range(class_num)}
cls_names = {i['id']:i['name'] for i in test_json['categories']}

for data in tqdm(test_loader):

    prediction_string = ''

    data = data[0]
    img = data['image'].copy()

    outputs = predictor(data['image'])['instances']

    targets = outputs.pred_classes.cpu().tolist()
    boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
    scores = outputs.scores.cpu().tolist()

    for target, box, score in zip(targets, boxes, scores):
        prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' '
                              + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')

        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_dict[target], 2)
        cv2.putText(img, str(cls_names[target])+'  '+str(score)[:4], (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[target], 1)

    cv2.imshow('', img)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()

    prediction_strings.append(prediction_string)
    file_names.append(data['file_name'].replace('../dataset/', ''))

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det3.csv'), index=None)
submission.head()