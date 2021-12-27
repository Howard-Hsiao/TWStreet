import pandas as pd

detection_data = pd.read_csv("./output/YOLOv5_detection.csv")
recognition_data = pd.read_csv("./output/paddle_recognition.csv", header=None, names=["imgName", "Label_data", "rec_confidence"])
output_data = pd.merge(detection_data, recognition_data, on="imgName")
output_data.to_csv(f"./output/merged_result.csv", index=False, header=False)