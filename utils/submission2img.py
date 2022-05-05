import cv2
import csv
import json
import numpy as np


with open('../dataset/submission.csv', 'r') as f:
    rdr = csv.reader(f)
    csv_data = list(rdr)

with open('../dataset/test.json', 'r') as f:
    test_json = json.load(f)

color_dict = {i: tuple(map(int, np.random.choice(range(256), size=3))) for i in range(10)}
cls_names = {i['id']:i['name'] for i in test_json['categories']}

dir_ = '../dataset/'
for line in csv_data[1:]:
    img = cv2.imread(dir_+line[1])

    ann = line[0].split()
    ann_new = [list(map(float,ann[i:i+6])) for i in range(0,len(ann),6)]
    for box in ann_new:
        img = cv2.rectangle(img, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), color_dict[box[0]], 2)
        cv2.putText(img, str(cls_names[int(box[0])]) + '  ' + str(box[1])[:4], (int(box[2]), int(box[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color_dict[box[0]], 1)

    cv2.imshow('', img)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
