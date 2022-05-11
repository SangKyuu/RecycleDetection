import os
import copy
import torch
import cv2
import json
import numpy as np
import argparse
import sys
import platform
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from ensemble_boxes import *
import detectron2
from detectron2.data import detection_utils as d_utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from models.CenterNet2.centernet.config import add_centernet_config

import torch.backends.cudnn as cudnn

from models.yolor.yolor_utils.google_utils import attempt_load
from models.yolor.yolor_utils.datasets import LoadStreams, LoadImages
from models.yolor.yolor_utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from models.yolor.yolor_utils.plots import plot_one_box
from models.yolor.yolor_utils.torch_utils import select_device, load_classifier, time_synchronized

from models.yolor.models.models import *
from models.yolor.yolor_utils.datasets import *
from models.yolor.yolor_utils.general import *


# Register Dataset
try:
    register_coco_instances('coco_trash_test', {}, './dataset/test.json', './dataset/')
except AssertionError:
    pass

def build_centernet2(cfg_path):
    # config 불러오기
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(cfg_path)

    # config 수정하기
    cfg.DATASETS.TEST = ('coco_trash_test',)
    # model
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def build_yolor():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='models/yolor/YOLOR_recycle_fintuned.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='dataset/test/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0, 1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='models/yolor/cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='models/yolor/data/recycle.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names

    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    opt.half = device.type != 'cpu'  # half precision only supported on CUDA

    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()

    if opt.half:
        model.half()  # to FP16
    return model, opt, device

# mapper - input data를 어떤 형식으로 return할지
def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = d_utils.read_image(dataset_dict['file_name'], format='BGR')

    dataset_dict['image'] = image

    return dataset_dict


if __name__ == '__main__':
    centernet_cfg_path = 'models/CenterNet2/configs/ensemble_config.yaml'
    Centernet, centernet_cfg = build_centernet2(centernet_cfg_path)
    Yolor, yolor_opt, device = build_yolor()
    # dataset = LoadImages(yolor_opt.source, img_size=yolor_opt.img_size, auto_size=64)

    # test loader
    c_test_loader = build_detection_test_loader(centernet_cfg, 'coco_trash_test', MyMapper)
    y_test_loader = LoadImages(yolor_opt.source, img_size=yolor_opt.img_size, auto_size=64)

    with open('./dataset/test.json', 'r') as f:
        test_json = json.load(f)

    # output 뽑은 후 sumbmission 양식에 맞게 후처리
    prediction_strings = []
    file_names = []
    DEBUG = False
    class_num = 10
    color_dict = {i: tuple(map(int,np.random.choice(range(256), size=3))) for i in range(class_num)}
    cls_names = {i['id']:i['name'] for i in test_json['categories']}

    for data, y_data in tqdm(zip(c_test_loader, y_test_loader)):

        prediction_string = ''

        data = data[0]
        img = data['image'].copy()  # display image

        y_img = torch.from_numpy(y_data[1]).to(device)  # yolor input
        y_img = y_img.half() if yolor_opt.half else y_img.float()  # uint8 to fp16/32
        y_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if y_img.ndimension() == 3:
            y_img = y_img.unsqueeze(0)

        outputs = Centernet(data['image'])['instances']

        pred = Yolor(y_img, augment=yolor_opt.augment)[0]
        pred = non_max_suppression(pred, yolor_opt.conf_thres, yolor_opt.iou_thres, classes=yolor_opt.classes, agnostic=yolor_opt.agnostic_nms)

        y_targets = []
        y_boxes = []
        y_scores = []

        gn = torch.tensor(data['image'].shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # YOLOR post_process
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to img_ size
                det[:, :4] = scale_coords(y_img.shape[2:], det[:, :4], img.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy).view(1, 4).view(-1)
                    xyxy = (xyxy / gn).tolist()
                    y_targets.append(int(cls.cpu().item()))
                    y_boxes.append(xyxy)
                    y_scores.append(conf.cpu().item())

        # CenterNet2 post_process
        c_targets = outputs.pred_classes.cpu().tolist()
        c_boxes = [(i.cpu()/gn).detach().tolist() for i in outputs.pred_boxes]
        c_scores = outputs.scores.cpu().tolist()

        t_targets = [c_targets, y_targets]
        t_boxes = [c_boxes, y_boxes]
        t_scores = [c_scores, y_scores]

        f_boxes, f_scores, f_labels = weighted_boxes_fusion(t_boxes, t_scores, t_targets, weights=[1,1], iou_thr=0.5, skip_box_thr=0.0001)
        # print(t_boxes, t_scores, t_targets)
        # print(f_boxes, f_scores, f_labels)

        for target, box, score in zip(f_labels, f_boxes, f_scores):
            if score > 0.1:
                box = [box[0]*data['image'].shape[0], box[1]*data['image'].shape[1], box[2]*data['image'].shape[0], box[3]*data['image'].shape[1]]
                prediction_string += (str(int(target)) + ' ' + str(score) + ' ' + str(box[0]) + ' '
                                      + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
                if DEBUG:
                    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_dict[target],
                                        2)
                    cv2.putText(img, str(cls_names[target]) + '  ' + str(score)[:4], (int(box[0]), int(box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[target], 1)

        if DEBUG:
            cv2.imshow('', img)
            if cv2.waitKey() == 27:
                cv2.destroyAllWindows()

        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace('./dataset/', ''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv('dataset/submission_ensemble_agnostice nms.csv', index=None)
    submission.head()
