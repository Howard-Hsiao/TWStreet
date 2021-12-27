python3 yolo_inference.py $1 $2
python3 -m paddle.distributed.launch --gpus '0' ./PaddleOCR/tools/infer_rec_to_csv.py -c ./configs/rec_chinese_common_train_v2.0.yml -o Global.checkpoints=$3 Global.infer_img=./temp
python3 merge_subresult.py