# Libraries importation
from ultralytics import YOLO

# Load pretrained model
model = YOLO("path to your trained best model")

# Train the model
results = model.val(data="path to your dataset",
        epochs=300,
        imgsz=640)
