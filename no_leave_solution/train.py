from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

if __name__ == '__main__': 

    data_train = model.train(data="C:/Users/savadogo_abdoul/Desktop/final_omdena_crop/test/data",
                            epochs = 100,
                            batch= 8,
                            imgsz=640)