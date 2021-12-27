import os
import json
import sys

train_folder = sys.argv[1]
json_folder = os.path.join(train_folder, "json")
label_folder = os.path.join(train_folder, "labels")

for file_name in os.listdir(json_folder):
    json_path = os.path.join(json_folder, file_name)
    label_path = os.path.join(label_folder, file_name)

    data = None
    with open(json_path, encoding="utf-8") as rfile:
        data = json.load(rfile)

    img_w = int(data["imageWidth"])
    img_h = int(data["imageHeight"])

    with open(label_path[:-5] + ".txt", 'w') as wfile:
        for shape in data["shapes"]:
            left  = img_w + 1
            right = -1
            top   = img_h + 1
            down  = -1

            if shape["group_id"] == 255:
                continue

            for x, y in shape["points"]:
                left  = min(left, x)
                right = max(right, x)
                top   = min(top, y)
                down  = max(down, y)

            centerX = (right + left)/2
            centerY = (down  + top)/2
            bbox_w  = right - left
            bbox_h  = down - top

            centerX /= img_w
            centerY /= img_h
            bbox_w  /= img_w
            bbox_h  /= img_h
            wfile.write(f"{shape['group_id']} {centerX} {centerY} {bbox_w} {bbox_h}\n")