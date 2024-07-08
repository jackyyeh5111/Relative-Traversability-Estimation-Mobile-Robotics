import cv2
import os
import re
import numpy as np
import pathlib

""" 
    rgb_ts_2021_11_09_16h15m31s_000002.tif,rgb_ts_2021_11_15_13h01m47s_000742.tif,424,240,"[0,126,104,1,151,41,0];[0,17,130,0,114,224,0];[1,396,201,1,253,207,0]"
"""

ROOT = '/Users/jackyyeh/SideProjects/Traversability-Estimation/W-RIZZ'
LABEL_PATH = os.path.join(ROOT, 'data/wayfast/train_labels.csv')
IMG_ROOT = os.path.join(ROOT, 'data/wayfast/rgb')
OUTPUT_DIR = os.path.join(ROOT, 'vis_annotation')

def imshow(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cnt = 0
with open(LABEL_PATH, 'r') as f:
    f.readline()
    for line in f.readlines():
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
        # remove certain characters
        pattern = r'[\"\[\]]' 
        for i, annot in enumerate(annot_list):
            re_annot = re.sub(pattern, '', annot)
            label1, img1_w, img1_h, label2, img2_w, img2_h, relation = re_annot.split(',')
            
            print ('label1:', label1)
            print ('img1_w:', img1_w)
            print ('img1_h:', img1_h)
            print ('label2:', label2)
            print ('img2_w:', img2_w)
            print ('img2_h:', img2_h)
            print ('relation:', relation)
            print ('img1.shape:', img1.shape)
            
            # draw
            radius = 5
            
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
            
            cv2.circle(canvas1 , (int(img1_w), int(img1_h)), radius, color1, cv2.FILLED)
            cv2.circle(canvas2 , (int(img2_w), int(img2_h)), radius, color2, cv2.FILLED)
            concat = np.hstack([canvas1, canvas2])
            
            # if cnt == 36:
            #     imshow("concat", concat)
                
            pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f'{cnt}.jpg')
            cv2.imwrite(out_path, concat)
            cnt += 1
            
            if cnt == 200:
                break
            
            
        