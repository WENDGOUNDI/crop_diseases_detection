# Libraries importation
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n-cls.pt")

# Train the model
results = model.train(data="path to your dataset",
        epochs=300,
        imgsz=640)
