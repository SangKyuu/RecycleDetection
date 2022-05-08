# RecycleDetection

## 1. Summary
[YOLOR](https://github.com/WongKinYiu/yolor), [CenterNet2](https://github.com/xingyizhou/CenterNet2) 모델의 앙상블을 통하여 모델을 설정하였습니다. 
학습된 YOLOR, CenterNet2 모델의 prediction 결과를 WBF(Weighted Box Fusion)를 통하여 후처리하여 최종 결과값을 뽑을 수 있도록 하였습니다.

YOLOR 및 CetnerNet2의 경우 [COCO dataset Leaderboard](https://paperswithcode.com/sota/object-detection-on-coco) 의 결과를 참고하여 선정하였습니다. 

- Ensemble을 위하여 2stage 모델과 1stage 모델 하나씩 선택

<br>

![image](https://user-images.githubusercontent.com/41942097/167284365-d45ad66b-2caa-4dd0-a891-f2b280e7abe4.png)

![image](https://user-images.githubusercontent.com/41942097/167284356-82b91ac9-ef04-4daa-bef2-8c19c7210aef.png)

<br>

- 각 모델은 따로따로 학습하여 각각 모델이 최고 성능을 갖는 파라미터를 찾을 수 있도록 하였습니다.

<br><br>

## 2. Experimetal results

### CenterNet2
- 기본 parameters : 
#### train NMS threshold
#### test NMS threshold
#### confidence threshold
#### augumentation

<br>

### YOLOR
- 기본 parameters :
#### 

<br>

### Ensemble
####

<br><br>

## 3. Instructions

- **docker** 환경구성
- /RecycleDetection/models/CenterNet2/docker/ 폴더 내의 파일로 docker 환경 구성

```
# 아래의 코드 실행
$ docker-compose build r_centernet2
$ docker-compose run r_centernet2
```

- **Train**
```
(r_centernet2)$ cd /home/RecycleDetection/models/CenterNet2/
(r_centernet2)$ python train_net.py  # train CenterNet2

(r_centernet2)$ cd /home/RecycleDetection/models/yolor/
(r_centernet2)$ ! python train.py --batch-size 6 --img 1280 1280 --data data/recycle.yaml --cfg cfg/yolor_p6.cfg --weights '/content/yolor_p6.pt' --device 0 --name yolor_p6 --hyp hyp.finetune.1280.yaml --epochs 30   # triain YOLOR
```

- **Infernece**
```
(r_centernet2)$ python Inference_Ensemble.py
```
## 4. Approach

- [EDA](./EDA.ipynb) : 파일 참고

- CeneterNet 모델 성능 개선 방향
  - 학습시 NMS threshold에 대한 설정 : 기존 trianing  NMS threshold가 0.9로 설정되어 있어 주변의 박스를 못 지우는 경향이 생김 -> threhold 0.7로 설정
  - prediction confidence값의 설정 : 0.3가 사람 눈에는 제일 좋아 보였으나 놓치는 bbox가 많음으로 0.1로 설정
  - lr 설정 : 기본값은 0.04로 설정되어 있었지만 GPU 메모리 부족으로 batch size가 줄어듬에 따라 lr 하향 조정 (0.01)

- YOLOR 모델 성능 개선 방향

- Ensemble 개선 방향

- Future work
  - class에 대한 data imbalance으로 인한 mAP 값 저하 : augumentation을 통한 상대적으로 부족한 class에 대한 데이터 확보 및 학습을 통하여 imbalance 문제 해결
  - Ensemble 모델 추가
  - class에 대한 NMS 수정 : 
      ```
      기존 NMS의 경우 예측된 class에 따라 분리하여 각각 class에 대하여 NMS를 실행함으로, 같은 물체를 다른 class로 예측 시 IOU값이 큼에도 불구하고 지워지지 않는 현상 발생 
      -> IOU가 0.9 이상인 bbox들에 대하여 class에 상관없이 NMS 진행할 수 있도록 코드 수정
      ```
