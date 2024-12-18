from ultralytics import YOLO

model = YOLO("C:/Users/savadogo_abdoul/Desktop/final_omdena_crop/Cassava_project/runs/classify/train/weights/best.pt")

if __name__ == '__main__': 

    data_train = model.val(data="C:/Users/savadogo_abdoul/Desktop/final_omdena_crop/Cassava_project/split_data",
                            epochs = 100,
                            batch= 8,
                            imgsz=640)