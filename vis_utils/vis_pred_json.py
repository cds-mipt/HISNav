import json
import numpy as np
import pycocotools.mask as mask_util
import mmcv
from scipy import ndimage
import cv2
from tqdm import tqdm

from config import dataset_meta


def vis(conf_threshold, pred_json, true_json, img_root, save_dir):
    with open(pred_json, 'rt', encoding='UTF-8') as f:
        predictions = json.load(f)

    with open(true_json, 'rt', encoding='UTF-8') as f:
        annotations = json.load(f)
        
    id2filename = dict()
    for img in annotations['images']:
        id2filename[img['id']] = img['file_name']
        
    for image_id in tqdm(id2filename):
        image_ids = np.asarray([x["image_id"] for x in predictions])

        current = (image_ids == image_id).nonzero()[0]

        scores = np.asarray([predictions[i]["score"] for i in current])
        pred_masks = np.asarray([predictions[i]["segmentation"] for i in current])
        pred_classes = np.asarray([predictions[i]["category_id"] for i in current])

        chosen = (scores > conf_threshold).nonzero()[0]
        scores = scores[chosen]
        pred_masks = pred_masks[chosen]
        pred_classes = pred_classes[chosen]
        
        pred_masks = [mask_util.decode(x)[:, :] for x in pred_masks]

        img_show = mmcv.imread(f'{img_root}/{id2filename[image_id]}')
        num_mask = len(pred_classes)
        
        seg_show = img_show.copy()
        for idx in range(num_mask):
            cur_mask = pred_masks[idx]
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            cur_mask_bool = cur_mask.astype(np.bool)
            cur_cate = pred_classes[idx]
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.5 \
                                      + np.array(dataset_meta[cur_cate][1], dtype=np.uint8) * 0.5

            cur_score = scores[idx]

            label_text = dataset_meta[cur_cate][0]
            # label_text += '|{:.02f}'.format(cur_score)
            # center
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green

        mmcv.imwrite(seg_show, f'{save_dir}/{id2filename[image_id]}')


if __name__ == '__main__':
    conf_threshold_ = 0.3
    model_ = 'blendmask'

    pred_json_ = '/home/adeshkin/projects/inst_seg/blend_mask/AdelaiDet/' \
                'work_dirs/blendmask_DLA_34_syncbn_4x_batch_8/model_0272219/inference/test_00_00.json'
    true_json_ = '/home/adeshkin/Desktop/habitat/annotations/test_00_00_instances.json'

    img_root_ = '/home/adeshkin/Desktop/habitat/images/test_00_00'
    save_dir_ = f'/home/adeshkin/Desktop/{model_}_test_00_00_thr_{conf_threshold_}'

    vis(conf_threshold_, pred_json_, true_json_, img_root_, save_dir_)
