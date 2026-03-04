from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"/public/home/panyuchen/codes/v9/add_modules/yolov9t-pose-p6-final4.yaml")
    model.train(
            data=r"/public/home/panyuchen/data/corn/02.dslr_17points_yolo/data.yaml",
            epochs=300,
            batch=2,
            project='PFLO_dslr',
            imgsz=1280,
            pretrained=False)