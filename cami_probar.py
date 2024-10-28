from ultralytics import YOLO
import numpy as np
#from prepro import process_image  

model = YOLO('./runs/classify/train2/weights/best.pt')

#cambiar ruta
image_path = r'C:\Users\Rosario\Videos\Camila\trashnet\data\dataset-resized\cami_project\papel_2.jpeg'

processed_image = image_path

if processed_image is not None:
    results = model(processed_image)
    #print(results)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    print(names_dict)
    print(probs)
    max_prob_index = np.argmax(probs)
    print(names_dict[max_prob_index])
else:
    print(f"No se pudo procesar la imagen: {image_path}")