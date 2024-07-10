
import cv2
import os
import re
import numpy as np
import pathlib
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import re
import pathlib
from matplotlib import pyplot as plt
from typing import Any, Dict, List
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_sample', '-n', type=int, required=True)
parser.add_argument('--model_type', type=str, default="vit_h")
parser.add_argument('--checkpoint', type=str, default="/home/jackyyeh/SideProjects/Traversability-Estimation/segment-anything/model_checkpoint/sam_vit_h_4b8939.pth")
parser.add_argument('--vis_output', '-v', action='store_true')
args = parser.parse_args()    
        
print('======= Initial arguments =======')
for key, val in vars(args).items():
    print(f"name: {key} => {val}")
print ()

# Set the random seed
np.random.seed(123)

ROOT = '/home/jackyyeh/SideProjects/Traversability-Estimation/W-RIZZ'
LABEL_PATH = os.path.join(ROOT, 'data/wayfast/train_labels.csv')
IMG_ROOT = os.path.join(ROOT, 'data/wayfast/rgb')
OUTPUT_PATH = os.path.join(ROOT, 'data/wayfast/sam_train_labels.csv')
OUTPUT_DIR = os.path.join(ROOT, 'vis_sam_annotation')

    
def load_sam_model():
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device="cuda")
    # output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    # amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam)
    return generator

def sample(binary_mask, n_sample):
    # sample N more samples
    points = np.column_stack(np.where(binary_mask))
    # Check if there are enough points to sample
    
    # make sure enough points to sample from the binary mask.
    n_sample = min(len(points), n_sample)
    
    # if len(points) < n_sample:
    #     raise ValueError("Not enough points to sample from the binary mask.")

    # Randomly sample 10 points
    sampled_indices = np.random.choice(points.shape[0], n_sample, replace=False)
    sampled_pts = points[sampled_indices] # (y, x)
    
    # Swap columns
    temp = np.copy(sampled_pts[:, 0])
    sampled_pts[:, 0] = sampled_pts[:, 1]
    sampled_pts[:, 1] = temp # (x, y)
    
    return sampled_pts

def get_mask(img, masks: List[Dict[str, Any]], target_pt) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    height, width, _ = img.shape
    canvas = np.zeros((height, width, 3), dtype=img.dtype)

    target_masks = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        
        color = np.random.random((1, 3))[0] * 255
        color = color.astype(np.int8)
        canvas[mask > 0, :] = color

        x, y = target_pt
        if mask[y][x]:
            target_mask = mask.copy()
            target_masks.append(target_mask)
    # out_img = cv2.hconcat((img, canvas))
    return canvas, target_masks

def select_target_mask(masks):
    num_pixel = np.inf
    target_mask = None
    for mask in masks:
        if  np.count_nonzero(mask) < num_pixel:
            num_pixel = np.count_nonzero(mask)
            target_mask = mask.copy()
    return target_mask

if __name__ == "__main__":

    generator = load_sam_model()

    """ 
        Our data from .csv is like:

        ex: 
            rgb_ts_2021_11_09_16h15m31s_000002.tif,rgb_ts_2021_11_15_13h01m47s_000742.tif,424,240,"[0,126,104,1,151,41,0];[0,17,130,0,114,224,0];[1,396,201,1,253,207,0]"
    """
    cnt = 0
    total_sam_labels = []
    with open(LABEL_PATH, 'r') as f:
        first_line = f.readline()
        
        # init output file and write first line
        with open(OUTPUT_PATH, 'w') as fw:
            fw.write(first_line)
        
        n_line = 0
        for line in tqdm(f.readlines()):
            
            print ("n_line:", n_line)
            n_line+=1
            
            line = line.strip()
            data = line.split(',')
            img_path1, img_path2, width, height = data[:4]
            
            img_path1 = os.path.join(IMG_ROOT, img_path1)
            img_path2 = os.path.join(IMG_ROOT, img_path2)
            
            if not os.path.isfile(img_path1) or not os.path.isfile(img_path2):
                continue
            
            print ('img_path1:', img_path1)
            print ('img_path2:', img_path2)
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            
            annots = ','.join(data[4:])
            annot_list = annots.split(';')
            
            """ 
                ex: "[0,126,104,1,151,41,0] -> 0,126,104,1,151,41,0
                
                
                ### definition of annotation (from dataset.py) ###
                for a in annotation_str.split(';'):
                    l = [int(s) for s in a[1:-1].split(',')]
                    scale_x = self._resolution[1] / width
                    scale_y = self._resolution[0] / height
                    # l[-1] == 0 => eq; l[-1] == 1 => latter is more; l[-1] == -1 => former is more
                    annotations.append((
                        # ensure that annotations are properly resized to desired resolution
                        l[0], # 0 if pt1 is in imgA, 1 if it's in imgB
                        max(0, min(round(scale_x * l[1]), self._resolution[1]-1)),
                        max(0, min(round(scale_y * l[2]), self._resolution[0]-1)),
                        l[3], # 0 if pt2 is in imgA, 1 if it's in imgB
                        max(0, min(round(scale_x * l[4]), self._resolution[1]-1)),
                        max(0, min(round(scale_y * l[5]), self._resolution[0]-1)),
                        l[6]
                    ))
            """
            
            # inference by sam
            sam_masks1 = generator.generate(img1)
            sam_masks2 = generator.generate(img2)
            
            # remove certain characters
            pattern = r'[\"\[\]]' 
            pairs = []
            for i, annot in enumerate(annot_list):
                re_annot = re.sub(pattern, '', annot)
                label1, img1_w, img1_h, label2, img2_w, img2_h, relation = re_annot.split(',')
                
                # print ('label1:', label1)
                # print ('img1_w:', img1_w)
                # print ('img1_h:', img1_h)
                # print ('label2:', label2)
                # print ('img2_w:', img2_w)
                # print ('img2_h:', img2_h)
                # print ('relation:', relation)
                # print ('img1.shape:', img1.shape)
                
                # draw
                radius = 3
                
                # decide color
                relation = int(relation)
                if relation == -1:
                    color1 = (255, 0, 0)
                    color2 = (0, 0, 255)
                elif relation == 0:
                    color1 = (0, 255, 255)
                    color2 = (0, 255, 255)
                elif relation == 1:
                    color1 = (0, 0, 255)
                    color2 = (255, 0, 0)
                
                # decide img
                canvas1 = img1.copy() if int(label1) == 0 else img2.copy()
                canvas2 = img1.copy() if int(label2) == 0 else img2.copy()
                cur_masks1 = sam_masks1.copy() if int(label1) == 0 else sam_masks2.copy()
                cur_masks2 = sam_masks1.copy() if int(label2) == 0 else sam_masks2.copy()

                pt1 = [int(img1_w), int(img1_h)]
                pt2 = [int(img2_w), int(img2_h)]

                # generate masks
                color_masks1, target_masks1 = get_mask(canvas1, cur_masks1, pt1)
                color_masks2, target_masks2 = get_mask(canvas2, cur_masks2, pt2)
                
                sampled_pts1 = []
                sampled_pts2 = []    
                target_mask1 = None
                target_mask2 = None
                try:
                    target_mask1 = select_target_mask(target_masks1)
                    target_mask2 = select_target_mask(target_masks2)
                    sampled_pts1 = sample(target_mask1, args.n_sample - 1)
                    sampled_pts2 = sample(target_mask2, args.n_sample - 1)
                except:
                    pass
                """ 
                    complete one data sample in csv:
                        rgb_ts_2021_11_09_16h15m31s_000002.tif,rgb_ts_2021_11_15_13h01m47s_000742.tif,424,240,
                        "[0,126,104,1,151,41,0];[0,17,130,0,114,224,0];[1,396,201,1,253,207,0]"
                        
                    pair example: 
                        [0,126,104,1,151,41,0]
                """
                pairs.append(f'[{re_annot}]') # include the original ground truth pair
                for pt1, pt2 in zip(sampled_pts1, sampled_pts2):
                    pair = f'[{label1},{pt1[0]},{pt1[1]},{label2},{pt2[0]},{pt2[1]},{relation}]'
                    pairs.append(pair)
                
                
                """ 
                    visualization
                """
                cv2.circle(canvas1, (int(img1_w), int(img1_h)), radius, color1, cv2.FILLED)
                cv2.circle(canvas2 , (int(img2_w), int(img2_h)), radius, color2, cv2.FILLED)
                ori_canvas1 = canvas1.copy()
                ori_canvas2 = canvas2.copy()
                if target_mask1 is None or target_mask2 is None:
                    print ('fail path:', img_path1)
                    print ('fail cnt:', cnt)
                    concat1 = np.hstack([ori_canvas1, canvas1, color_masks1])
                    concat2 = np.hstack([ori_canvas2, canvas2, color_masks2])
                    output_dir = os.path.join(OUTPUT_DIR, "fail")
                else:
                    # draw circle
                    for pt in sampled_pts1:
                        cv2.circle(canvas1 , pt, radius, color1, cv2.FILLED)
                    for pt in sampled_pts2:
                        cv2.circle(canvas2 , pt, radius, color2, cv2.FILLED)

                    # target_mask1 = cv2.cvtColor(target_mask1.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    # target_mask2 = cv2.cvtColor(target_mask2.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    # concat1 = np.hstack([ori_canvas1, canvas1, target_mask1 * 255, color_masks1])
                    # concat2 = np.hstack([ori_canvas2, canvas2, target_mask2 * 255, color_masks2])
                    # output_dir = os.path.join(OUTPUT_DIR, "success")

                    ######
                    target_masks1 = [255 * cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR) for mask in target_masks1]
                    target_masks2 = [255 * cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR) for mask in target_masks2]
                    concat1 = np.hstack([canvas1, color_masks1] + target_masks1)
                    concat2 = np.hstack([canvas2, color_masks2] + target_masks2)
                    
                    width1 = concat1.shape[1]
                    width2 = concat2.shape[1]
                    if (width1 > width2):    
                        pad = np.zeros_like(concat1)
                        pad[:, :width2, :] = concat2
                        concat = np.vstack([concat1, pad])
                    else:
                        pad = np.zeros_like(concat2)
                        pad[:, :width1, :] = concat1
                        concat = np.vstack([pad, concat2])
                        
                    output_dir = os.path.join(OUTPUT_DIR, "success")

                if args.vis_output:
                    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
                    out_path = os.path.join(output_dir, f'{cnt}.jpg')
                    cv2.imwrite(out_path, concat)
                    cnt += 1
                
            # num_pairs_in_a_line = 3
            # start = 0
            # for i in range(len(pairs)):
            
            tmp_pairs = ';'.join(pairs)
            total_pairs = f'"{tmp_pairs}"' # "[0,126,104,1,151,41,0];[0,17,130,0,114,224,0];[1,396,201,1,253,207,0]"
            after_sample = data[:4] + [total_pairs]
            after_sample = ','.join(after_sample)
            total_sam_labels.append(after_sample)
        
            # print (after_sample)
            # input()
            
            # output
            with open(OUTPUT_PATH, 'a') as fw:
                # f.write(first_line)
                fw.write(f'{after_sample}\n')
            
            