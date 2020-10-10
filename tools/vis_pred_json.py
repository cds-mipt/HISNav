import json
import numpy as np
import pycocotools.mask as mask_util
import mmcv
from scipy import ndimage
import cv2
from tqdm import tqdm

colors = {1: [119, 11, 32],
 2: [0, 0, 142],
 3: [0, 0, 230],
 4: [106, 0, 228],
 5: [0, 60, 100],
 6: [0, 80, 100],
 7: [0, 0, 70],
 8: [0, 0, 192],
 9: [250, 170, 30],
 10: [100, 170, 30],
 11: [220, 220, 0],
 12: [175, 116, 175],
 13: [250, 0, 30],
 14: [165, 42, 42],
 15: [255, 77, 255],
 16: [0, 226, 252],
 17: [182, 182, 255],
 18: [0, 82, 0],
 19: [120, 166, 157],
 20: [110, 76, 0],
 21: [174, 57, 255],
 22: [199, 100, 0],
 23: [72, 0, 118],
 24: [255, 179, 240],
 25: [0, 125, 92],
 26: [209, 0, 151],
 27: [188, 208, 182],
 28: [0, 220, 176],
 29: [255, 99, 164],
 30: [92, 0, 73],
 31: [133, 129, 255],
 32: [78, 180, 255],
 33: [0, 228, 0],
 34: [174, 255, 243],
 35: [45, 89, 255],
 36: [134, 134, 103],
 37: [145, 148, 174],
 38: [255, 208, 186],
 39: [197, 226, 255],
 40: [171, 134, 1]}

dataset_id_map = {1: 'wall',
            2: 'floor',
            3: 'chair',
            4: 'door',
            5: 'table',
            6: 'picture',
            7: 'cabinet',
            8: 'cushion',
            9: 'window',
            10: 'sofa',
            11: 'bed',
            12: 'curtain',
            13: 'chest_of_drawers',
            14: 'plant',
            15: 'sink',
            16: 'stairs',
            17: 'ceiling',
            18: 'toilet',
            19: 'stool',
            20: 'towel',
            21: 'mirror',
            22: 'tv_monitor',
            23: 'shower',
            24: 'column',
            25: 'bathtub',
            26: 'counter',
            27: 'fireplace',
            28: 'lighting',
            29: 'beam',
            30: 'railing',
            31: 'shelving',
            32: 'blinds',
            33: 'gym_equipment',
            34: 'seating',
            35: 'board_panel',
            36: 'furniture',
            37: 'appliances',
            38: 'clothes',
            39: 'objects',
            40: 'misc'}

def main():
    conf_threshold = 0.3
    model = 'blendmask'
    root = '/home/adeshkin/projects/inst_seg'
    pred_json = 'blend_mask/AdelaiDet/work_dirs/blendmask_DLA_34_syncbn_4x_batch_8/model_0272219/inference/test_00_00.json'
    
    with open(f'{root}/{pred_json}', 'rt', encoding='UTF-8') as annotations:
        predictions = json.load(annotations)
        
    with open('/home/adeshkin/Desktop/habitat/annotations/test_00_00_instances.json', 'rt', encoding='UTF-8') as annotations:
        test_00_00 = json.load(annotations)
        
        
        
    id2filename = dict()
    for img in test_00_00['images']:
        id2filename[img['id']] = img['file_name']
        
    for image_id in tqdm(id2filename):
        image_ids = np.asarray([x["image_id"] for x in predictions])

        current = (image_ids == image_id).nonzero()[0]
        current_image_ids = image_ids[current]

        scores = np.asarray([predictions[i]["score"] for i in current])
        pred_masks = np.asarray([predictions[i]["segmentation"] for i in current])
        pred_classes = np.asarray([predictions[i]["category_id"] for i in current])

        chosen = (scores > conf_threshold).nonzero()[0]
        scores = scores[chosen]
        pred_masks = pred_masks[chosen]
        pred_classes = pred_classes[chosen]
        
        pred_masks = [mask_util.decode(x)[:, :] for x in pred_masks]
        img_show = mmcv.imread(f'/home/adeshkin/Desktop/habitat/images/test_00_00/{id2filename[image_id]}')
        num_mask = len(pred_classes)
        
        seg_show = img_show.copy()
        for idx in range(num_mask):
            cur_mask = pred_masks[idx]
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            cur_mask_bool = cur_mask.astype(np.bool)
            cur_cate = pred_classes[idx]
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.5 + np.array(colors[cur_cate], dtype=np.uint8) * 0.5

            cur_score = scores[idx]

            label_text = dataset_id_map[pred_classes[idx]]
            # label_text += '|{:.02f}'.format(cur_score)
            # center
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green

        save_dir = f'/home/adeshkin/Desktop/predictions_test_00_00/{model}_new_test_00_00_thr_{conf_threshold}'
        
        mmcv.imwrite(seg_show, f'{save_dir}/{id2filename[image_id]}')

if __name__ == '__main__':
    main()
