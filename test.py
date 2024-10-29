# Libraries importation
from ultralytics import YOLO

# Load pretrained model
model = YOLO("./runs/classify/train4/weights/best.pt") # adjust the path to point to your best trained model

# Train the model
results = model.test(data="path to your data",
        epochs=300,
        imgsz=640)
