import json
import csv

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def pair_images(pred_json, test_json, th=0.2):
    csv_data = [['',''] for i in range(len(test_json['images']))]
    for pred in pred_json:
        pred['bbox'] = [pred['bbox'][0], pred['bbox'][1], pred['bbox'][0]+pred['bbox'][2], pred['bbox'][1]+pred['bbox'][3]]
        if len(csv_data[pred['image_id']][1]):
            prev_line = csv_data[pred['image_id']][0]
            line = ' ' + ' '.join(map(str, [pred['category_id']] + [pred['score']] + pred['bbox'])) if pred['score'] >th else ''
            prev_line += line
            csv_data[pred['image_id']][0] = prev_line
        else:
            line = ' '.join(map(str, [pred['category_id']] + [pred['score']] + pred['bbox'])) if pred['score'] >th else ''
            csv_data[pred['image_id']] = [line, test_json['images'][pred['image_id']]['file_name']]
    csv_data.insert(0, ['PredictionString', 'image_id'])
    return csv_data


if __name__ == '__main__':
    prediction_json = load_json('/home/server1/workdir/project_ing/SK_temp/AIStages/RecycleDetection/models/CenterNet2/output/CenterNet2/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST/inference_recycle_test/coco_instances_results.json')
    test_json = load_json('../dataset/test.json')

    csv_data = pair_images(prediction_json, test_json)

    with open('../dataset/submission_train_test_nms05_c02.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(csv_data)
