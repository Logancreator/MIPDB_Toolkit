from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"/public/home/panyuchen/codes/v9/add_modules/yolov9t-pose-p6-final4.yaml")
    model.train(
        data="/public/home/panyuchen/data/corn/02.uav_17points_yolo/data.yaml",
        epochs=150,
        batch=4,
        imgsz=960,
        lr0=0.002,
        lrf=0.01,
        patience=20,
        close_mosaic=30,
        pretrained=False,
        project="PFLO_uav")