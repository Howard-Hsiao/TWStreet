import json
import os
from PIL import Image
import cv2
import numpy as np
import sys

write = ''
json_root = sys.argv[1]
img_root = sys.argv[2]
output_dir = sys.argv[3]

for file_name in os.listdir(json_root): 
    # read the json
    with open(os.path.join(json_root, file_name), 'r') as j:
        contents = json.loads(j.read())

    bbox_list = contents['shapes']
    img = Image.open(os.path.join(img_root, file_name[:-5]+'.jpg'))  
    cnt = 0

    for ele in bbox_list:
        if ele['label'] == '###' or ele['label'] == '':
            continue
        image_path = '%s/%s_%02d.jpg'%(output_dir, file_name[:-5], cnt)
        bbox = ele['points']  
        xs = [bbox[0][0],bbox[3][0],bbox[1][0],bbox[2][0]]
        ys = [bbox[0][1],bbox[3][1],bbox[1][1],bbox[2][1]]
        width = bbox[1][0] - bbox[0][0]
        height = bbox[2][1] - bbox[1][1]
        
        # convert the bbox to rectangle
        img_ = cv2.imread(os.path.join(img_root, file_name[:-5]+'.jpg'))
        pts1 = np.float32(bbox)
        pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img_, M, (width, height))
        
        # save the images
        if width < height:  # vertical
          dst = np.rot90(dst)
          cv2.imwrite(image_path, dst) 
          
        else:
          cv2.imwrite(image_path, dst) 
        cnt += 1
                    
        write += image_path+'\t'+ele['label']+'\n'
	
	
f = open('%s.txt'%(output_dir), 'w')
f.writelines(write)
f.close()	
	
      