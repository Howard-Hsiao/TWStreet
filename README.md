# TWStreet competition
## Membership
- 林語萱
- 蕭昀豪
- 陳姵如

## Installation
We used python3.6.9 to implement this solution. 
You can use the following script to install the required packages. 
- requirements.txt is "pip list" result of our virtualenv, if you want to check the specific package version we used, this file is what you want.
```{bash}
pip install -r requirements.txt
```
- inference_requirements.txt
This file contains the minimum requirements which allow you to execute our command in the environments of python3.6.9. 
```{bash}
pip install -r inference_requirements.txt
```

### Reminder
If you want to train your model and just use inference_requirements.txt to install the required package, that might be insufficient. You should further go to the home page of [YOLOv5](https://github.com/ultralytics/yolov5) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to see their detailed procedure. 

## Download Models
```{wget}
bash get_models.sh
```

## Train
### YOLOv5 (detection model)
#### PreProcessing
In order to satisfy the requirement of YOLOv5, we should first adjust the format of the ground truth. 
```
python3 prepare_YOLOv5_format.py <train_folder>
```

#### Train
using YOLOv5 api
```
python train.py --img 1280 --batch 6 --epochs 100 --data ./configs/TWStreet.yaml --weights yolov5l6.pt --device 1,2,3
```
### PaddleOCR (recognition model)
#### PreProcessing
In order to satisfy the requirement of PaddleOCR, we should first adjust the format of the ground truth. 
```
python3 prepare_rec_train_format.py <train_json_folder> <train_image_folder> <output_folder>
```
Then we should modify the file paths in 'PaddleOCR/configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml' to our own file paths, including the 'data_iter' and 'label_file_list' of the Train dataset and Eval dataset. 
#### Train
```
python3 -m paddle.distributed.launch --gpus '0' PaddleOCR/tools/train.py -c PaddleOCR/configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml
```

## Inference
You can use bash inference.sh to get the prediction of your images. 
Before you start, please make sure: 
- Your repository contains **"./temp/"** and **"./output"** folder. 
    - **"./temp/"** folder is used to store the images produced by YOLOv5. Once you get the final prediction result, you can remove it. 
    - **"./output"** folder is used to store the inference result of YOLOv5 and PaddleOCR, and the final prediction result of the entire package.

## **Reminder**
Owing to the limitation of the torch.hub.load() function, if you want to run the above inference script, you should ensure that your device has at least 1 gpu and there is enough memery left on the gpu of index 0. 

#### Argument
- $1 The folder containing images interested
- $2 The detection model path
- $3 The recognition model path

#### Example
```{bash}
bash inference.sh ./sample ./models/detection.pt  ./models/rec_chinese_common_v2.0/iter_epoch_29
```

### Post Processing
The adjustment of threshold to the recognition confidence is important in our implementation, so we seperate the post processing part from inference in order to easily adjust it. 

```{python}
python3 post_processing.py <threshold> <input.csv> <output.csv>
# example: 
# python3 post_processing.py 0.85 ./output/merged_result.csv ./output/final_result_0-85.csv 
```
