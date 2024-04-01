from ultralytics import YOLO

# import torch

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
# print(torch.cuda.devices())

# Train the model
results = model.train(
    data="/home/GPU/tsutar/home_gtl/Traffic_sign_detection/datasets/LISA_yolo/data.yaml",
    epochs=100,
    device=1,
    batch=-1,
    patience=5,
)