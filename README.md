# 재활용 품목 분류를 위한 Object Detection

## 1. Summary

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/167637173-6780bac2-d183-4550-9571-0078adcdf41d.png" width="730" height="350"></p>

- [YOLOR](https://github.com/WongKinYiu/yolor), [CenterNet2](https://github.com/xingyizhou/CenterNet2) 모델의 앙상블을 통하여 모델을 설정하였습니다. 
학습된 YOLOR, CenterNet2 모델의 prediction 결과를 WBF(Weighted Box Fusion)를 통하여 후처리하여 최종 결과값을 뽑을 수 있도록 하였습니다.
(다른 결과를 추론하기 위하여 1 stage Detector, 2 stage Detector 각각 선택)

### Training Elements

#### CenterNet2
|Learning Rate|Augumentation|Train NMS threshold|Imgae Size(train)|Imgae Size(test)|epoch|
|---|---|---|---|---|---|
|0.01|crop, brightness, contrast|0.7|(640,640)|(640,640)|30|

<br>

#### YOLOR
|Learning Rate|Augumentation|Train NMS threshold|Imgae Size(train)|Imgae Size(test)|epoch|
|---|---|---|---|---|---|
|0.01|flip, mosaic|0.5|(1024,1024)|(1024,1024)|50|

<br><br>

## 2. Experimetal results

### A. CenterNet2

#### train NMS threshold
(초기 학습 결과를 확인하였을 때 NMS 처리에 따른 지워지지 않는 box들의 존재를 확인하고 th를 조정해보았습니다.)

|Training NMS th|Test NMS th|mAP50|
|---|---|---|
|0.9|0.9|55.54|
|0.7|0.7|58.75|
|0.7|0.5|**59.67**|
|0.5|0.5|56.35|

#### Soft-NMS
- Soft-NMS로 NMS를 교체하였을 때, mAP50이 32.80으로 현저히 낮은 성능을 보였습니다.


#### confidence threshold
(confidence가 높을수록 사람 눈에는 더욱 좋은 결과로 느껴졌지만, Precision 결과상으로는 낮으수록 더 많은 box들이 남아 더 좋은 결과를 보였습니다.)
|Confidence th|mAP50|
|---|---|
|0.3|54.98|
|0.2|57.15|
|0.1|**58.75**|


#### augumentation
|Train|Test|mAP50|
|---|---|---|
|A type|None|55.54|
|B type|B type|51.27|
|A type|None|**57.10**|

- A type: Random Crop
- B type: Random Crop, Random Brightness, Random Contrast

-> 학습에만 augmentation을 진행하는 것이 최고로 좋은 결과를 보였습니다. 

<br>

### B. YOLOR

#### Image size
|Imgae size| mAP50|
|---|---|
|1024|52.67|
|1280|**54.28**|

#### Confidence
|Confidence|mAP50|
|---|---|
|0.1|52.67|
|0.05|**54.28**|

-> YOLOR의 경우 epoch 30으로는 부족하여 50으로 증가했을 때 mAP50 57.58 기존 54.28보다 크게 증가하였습니다.

<br>

### C. Ensemble
#### WBF confidence

|Confidence|mAP50|
|---|---|
|0.1|61.50|
|0.0001|**61.70**|

#### WBF model weights

|CenterNet:YOLOR|mAP50|
|---|---|
|1:1|**61.70**|
|2:1|61.49|

-> WBF의 경우 CenterNet2 와 YOLOR의 가중치를 동일하게 설정하는 것이 좋은 성능을 보였고, wbf를 통하여 Ensemble을 진행함으로써 CenterNet2의 최고 점수인 59.67를 넘을 수 있었습니다. 
- WBF의 경우 모델의 구조가 다를수록 효과가 크다는 것을 논문을 통하여 확인하고, 2 stage Detector인 CenterNet2 와 1 stage Detector인 YOLOR을 선택하여 진행하였습니다.


<br><br>

## 3. Instructions
- git clone
```
$ git clone https://github.com/SangKyuu/RecycleDetection/
```

- dataset 다운로드 
```
$cd ./RecycleDetection/
$ wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000086/data/data.tar.gz
$ tar -xvf data.tar.gz
```

- 학습된 model Weight 다운로드
```
$ wget https://drive.google.com/file/d/1-fAVUb7CWd-DaP4ijMerZjNCLVlN8ZYi/view?usp=sharing  # CenterNet weight (finetuned)
$ wget https://drive.google.com/file/d/1iL1tB7h1s_PcXwvbBd-ETvaAjJRm-6nc/view?usp=sharing  # YOLOR weight (finetuned)
$ mv YOLOR_recycle_fintuned.pt RecycleDetection/models/yolor/YOLOR_recycle_fintuned.pt
$ mv CenterNet2_recycle_finetuned.pth RecycleDetection/models/CenterNet2/CenterNet2_recycle_finetuned.pth
```

- **docker** 환경구성
- /RecycleDetection/models/CenterNet2/docker/ 폴더 내의 파일로 docker 환경 구성

```
# 아래의 코드 실행
$ cd RecycleDetection/models/CenterNet2/docker/
$ docker-compose build r_centernet2
$ docker-compose run r_centernet2
```

- **Train**
```
# inside docker container
(r_centernet2)$ cd /home/RecycleDetection/models/CenterNet2/
(r_centernet2)$ python train_net.py  # train CenterNet2

(r_centernet2)$ cd /home/RecycleDetection/
(r_centernet2)$ python models/yolor/make_yolor_data.py
(r_centernet2)$ python train.py --batch-size 6 --img 1280 1280 --data models/yolor/data/recycle.yaml --cfg models/yolor/cfg/yolor_p6.cfg --weights 'models/yolor/content/yolor_p6.pt' --device 0 --name yolor_p6 --hyp models/yolor/data/hyp.finetune.1280.yaml --epochs 50   # triain YOLOR
```

- **Infernece**
```
(r_centernet2)$ python inference_ensemble.py
```

<br><br>

## 4. Approach

- [EDA](./EDA.ipynb) : 파일 참고
  - EDA 파일을 통하여 확인하였을 때 class에 대한 imbalance가 컸으므로 focal loss를 사용하는 CenterNet2 모델을 선정하였습니다.

- CeneterNet 모델 성능 개선 방향
  - 학습시 NMS threshold에 대한 설정 : 기존 trianing  NMS threshold가 0.9로 설정되어 있어 주변의 박스를 못 지우는 경향이 생김 -> threhold 0.7로 설정하였습니다.
  - prediction confidence값의 설정 : 0.3가 사람 눈에는 제일 좋아 보였으나 놓치는 bbox가 많음으로 0.1로 설정하였습니다.
  - lr 설정 : 기본값은 0.04로 설정되어 있었지만 GPU 메모리 부족으로 batch size가 줄어듬에 따라 lr (0.01) 하향 조정하였습니다.

- YOLOR 모델 성능 개선 방향
  - 30 epoch으로 학습 시 train에 대한 데이터가 충분히 학습되지 않음을 확인하였습니다. 따라서 epoch을 50으로 증가시켜 학습을 진행하였습니다.
  - YOLOR을 학습시킨 기본 이미지 사이즈인 1280이 아닌 1024로 학습하였을 때, 정확도가 낮을것을 판단할 수 있었습니다.
 
- Ensemble 개선 방향
  - 서로 구조가 상이한 모델 2개를 통하여 ensemble하여 정밀도를 올리고자 하였습니다.
  - ensemble은 각 모델의 prediction값을 wbf을 통하여 후처리 후 최종값을 출력하도록 하였습니다. 이때 각 모델의 비중은 1:1로 설정함으로써 각 모델이 동등하게 예측할 수 있도록 하였습니다. 
  - 
- Future work
  - class에 대한 data imbalance으로 인한 mAP 값 저하 : augumentation을 통한 상대적으로 부족한 class에 대한 데이터 확보 및 학습을 통하여 imbalance 문제 해결하였습니다.
  - Ensemble 모델 추가 (ex Swin-T, 1 stage 계열의 모델)
  - class에 대한 NMS 수정 : 
      ```
      기존 NMS의 경우 예측된 class에 따라 분리하여 각각 class에 대하여 NMS를 실행함으로, 같은 물체를 다른 class로 예측 시 IOU값이 큼에도 불구하고 지워지지 않는 현상 발생 
      -> IOU가 0.9 이상인 bbox들에 대하여 class에 상관없이 NMS 진행할 수 있도록 코드 수정하면 성능이 오를것이라 생각합니다.
      ```
