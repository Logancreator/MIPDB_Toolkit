from ultralytics import YOLO



if __name__ == "__main__":
   # # net = YOLO(r'/public/home/panyuchen/codes/v9baseline/ultralytics-main/all/v8x-pose-1280/train/weights/best.pt')
   # net = YOLO(r'/public/home/panyuchen/codes/PFLO-UAV/PFLO_uav_unpretrained_div1/train/weights/best.pt')
   # output = net(r'/public/home/panyuchen/data/corn/uav/yolo_dataset_div1/images/val',
   #               save=True,
   #               show_labels=False,
   #               show_boxes=False,
   #               #save_txt=True,
   #               project='PFLO_uav_unpretrained_div1',
   #               )
   # net = YOLO(r'/public/home/panyuchen/codes/PFLO-UAV/PFLO_uav_pretrained_div1/train/weights/best.pt')
   # output = net(r'/public/home/panyuchen/data/corn/uav/yolo_dataset_div1/images/val',
   #             save=True,
   #             show_labels=False,
   #             show_boxes=False,
   #             # save_txt=True,
   #             project='PFLO_uav_pretrained_div1',
   #             )
   # net = YOLO(r'/public/home/panyuchen/codes/v9baseline/ultralytics-main/all/v8x-pose-1280/train/weights/best.pt')
   # net = YOLO(r'/public/home/panyuchen/codes/PFLO-UAV/PFLO_uav/train/weights/best.pt')
   # output = net(r'/public/home/panyuchen/data/corn/02.uav_17points_yolo/images/val',
   #               save=True,
   #               show_labels=False,
   #               show_boxes=False,
   #               save_txt=True,
   #               project='PFLO_uav_unpretrained_div1',
   #               )
   
   # net = YOLO(r'/public/home/panyuchen/codes/PFLO-UAV/PFLO_dslr/train/weights/best.pt')
   # output = net(r'/public/home/panyuchen/data/corn/02.dslr_17points_yolo/images/val',
   #             save=True,
   #             show_labels=False,
   #             show_boxes=False,
   #             save_txt=True,
   #             project='PFLO_dslr_pretrained_div1',
   #             )
   # net = YOLO(r'/public/home/panyuchen/codes/PFLO-UAV/PFLO_dslr/train/weights/best.pt')
   # output = net(r'/public/home/panyuchen/data/corn/02.dslr_17points_yolo/images/hzau_data2',
   #             save=True,
   #             show_labels=False,
   #             show_boxes=False,
   #             save_txt=True,
   #             project='PFLO_dslr_hzau_data'
   #             )
   net = YOLO(r'/public/home/panyuchen/codes/PFLO-UAV/PFLO_uav/train/weights/best.pt')
   output = net(r'/public/home/panyuchen/data/corn/02.dslr_17points_yolo/images/hzau_data3',
               save=True,
               show_labels=False,
               show_boxes=False,
               save_txt=True,
               project='PFLO_dslr_hzau_data'
               )