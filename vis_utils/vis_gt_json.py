import json
import numpy as np
import pycocotools.mask as mask_util
import mmcv
from scipy import ndimage
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw

from config import dataset_meta


def vis(json_path, height, width, img_root, save_dir):
    with open(json_path, 'rt', encoding='UTF-8') as f:
        json_data = json.load(f)

    id2filename = dict()
    for img in json_data['images']:
        id2filename[img['id']] = img['file_name']

    annotations = json_data['annotations']

    for image_id in tqdm(id2filename):
        image_ids = np.asarray([x["image_id"] for x in annotations])

        current = (image_ids == image_id).nonzero()[0]
        masks = []
        classes = []
        for i in current:
            masks_ = annotations[i]["segmentation"]
            for polygon in masks_:
                img = Image.new('L', (width, height), 0)

                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                mask = np.array(img)
                masks.append(mask)
                classes.append(annotations[i]["category_id"])
        masks = np.asarray(masks)

        img_show = mmcv.imread(f'{img_root}/{id2filename[image_id]}')
        num_mask = len(masks)

        class_text_num = dict()
        for i in range(1, 41):
            class_text_num[i] = 0

        seg_show = img_show.copy()
        for idx in range(num_mask):
            cur_mask = masks[idx]
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            cur_mask_bool = cur_mask.astype(np.bool)
            cur_cate = classes[idx]
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.5 \
                                      + np.array(dataset_meta[cur_cate][1], dtype=np.uint8) * 0.5

            class_text_num[cur_cate] += 1
            if class_text_num[cur_cate] > 4:
                label_text = ''
            else:
                label_text = dataset_meta[cur_cate][0]

            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))

        mmcv.imwrite(seg_show, f'{save_dir}/{id2filename[image_id]}')


if __name__ == '__main__':
    json_path_ = '/home/adeshkin/Desktop/habitat/annotations/test_00_00_instances.json'
    height_ = 320
    width_ = 640
    img_root_ = '/home/adeshkin/Desktop/habitat/images/test_00_00'
    save_dir_ = f'/home/adeshkin/Desktop/gt_test_00_00___'
    vis(json_path_, height_, width_, img_root_, save_dir_)
