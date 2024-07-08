import cv2
import os
import re
import numpy as np
import pathlib

ROOT = '/Users/jackyyeh/SideProjects/Traversability-Estimation/W-RIZZ'
LABEL_PATH = os.path.join(ROOT, 'data/wayfast/valid_labels.csv')
IMG_ROOT = os.path.join(ROOT, 'data/wayfast/rgb')
OUTPUT_DIR = os.path.join(ROOT, 'vis_prediction')
PRED_PATH = os.path.join(ROOT, 'predictions.npy')

def imshow(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predictions = np.load(PRED_PATH) # (1655, 2, 240, 424)
idx = 0
with open(LABEL_PATH, 'r') as f:
    f.readline()
    for line in f.readlines():
        line = line.strip()
        
        data = line.split(',')
        raw_img_path1, raw_img_path2, width, height = data[:4]
        raw_img_paths = [raw_img_path1, raw_img_path2]
        
        for i in range(2):
            raw_img_path = raw_img_paths[i]
            img_path = os.path.join(IMG_ROOT, raw_img_path)
        
            if not os.path.isfile(img_path):
                continue
            
            print ('img_path:', img_path)
            img = cv2.imread(img_path)

            # Apply colormap to the heatmap
            prediction = (predictions[idx][i] * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(prediction, cv2.COLORMAP_JET)
            
            # Overlay the heatmap on the image
            overlayed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            concat = np.hstack([img, overlayed_img])
            # imshow("concat", concat)
            
            pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            fname = raw_img_path.strip('.tif')
            out_path = os.path.join(OUTPUT_DIR, f'{fname}.jpg')
            if os.path.isfile(out_path):
                continue
            
            cv2.imwrite(out_path, concat)
            
        idx += 1