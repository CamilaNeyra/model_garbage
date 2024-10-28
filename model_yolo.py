import torch
import multiprocessing
from ultralytics import YOLO




def main():
    multiprocessing.set_start_method('spawn', force=True)
    
    model = YOLO('yolov8m-cls.pt') 
    #cambiar la ruta de tu dta
    model.train(data=r'C:\Users\Rosario\Videos\Camila\trashnet\data\dataset-resized\cami_project\data_nueva', epochs=100, imgsz=320, batch=4,patience=15)

if __name__ == '__main__':
    main()

