from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

if __name__ == '__main__': 

    data_train = model.train(data="C:/Users/savadogo_abdoul/Desktop/test_gpu/split_data/",
                            epochs = 100,
                            batch= 8,
                            imgsz=640)