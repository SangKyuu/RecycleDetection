import os
import copy
import torch
import cv2
import json
import numpy as np
import argparse
import platform
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from ensemble_boxes import *
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

from .models.CenterNet2.centernet.config import add_centernet_config

import torch.backends.cudnn as cudnn

from .models.yolor.utils.google_utils import attempt_load
from .models.yolor.utils.datasets import LoadStreams, LoadImages
from .models.yolor.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from .models.yolor.utils.plots import plot_one_box
from .models.yolor.utils.torch_utils import select_device, load_classifier, time_synchronized

from .models.yolor.models.models import *
from .models.yolor.utils.datasets import *
from .models.yolor.utils.general import *


# Register Dataset
try:
    register_coco_instances('coco_trash_test', {}, '../../dataset/test.json', '../../dataset/')
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
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()

    if half:
        model.half()  # to FP16
    return model, opt

# mapper - input data를 어떤 형식으로 return할지
def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    dataset_dict['image'] = image

    return dataset_dict


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


if __name__ == '__main__':
    centernet_cfg_path = 'output/CenterNet2/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST(nms0.7test)/config.yaml'
    Centernet, centernet_cfg = build_centernet2(centernet_cfg_path)
    Yolor, yolor_opt = build_yolor()
    # names = load_classes(names)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # test loader
    test_loader = build_detection_test_loader(centernet_cfg, 'coco_trash_test', MyMapper)

    with open('../../dataset/test.json', 'r') as f:
        test_json = json.load(f)

    # output 뽑은 후 sumbmission 양식에 맞게 후처리
    prediction_strings = []
    file_names = []
    DEBUG = False
    class_num = 10
    color_dict = {i: tuple(map(int,np.random.choice(range(256), size=3))) for i in range(class_num)}
    cls_names = {i['id']:i['name'] for i in test_json['categories']}

    for data in tqdm(test_loader):

        prediction_string = ''

        data = data[0]
        img = data['image'].copy()

        outputs = Centernet(data['image'])['instances']

        img_ = data['image']/255.0
        if img_.ndimension() == 3:
            img_ = img_.unsqueeze(0)

        pred = Yolor(img, augment=yolor_opt.augment)[0]
        pred = non_max_suppression(pred, yolor_opt.conf_thres, yolor_opt.iou_thres, classes=yolor_opt.classes, agnostic=yolor_opt.agnostic_nms)

        # YOLOR post_process
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s # original image

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

        # CenterNet2 post_process
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()

        # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        for target, box, score in zip(targets, boxes, scores):
            if score > 0.1:
                prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' '
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
        file_names.append(data['file_name'].replace('../../dataset/', ''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(centernet_cfg.OUTPUT_DIR, f'submission_det3.csv'), index=None)
    submission.head()