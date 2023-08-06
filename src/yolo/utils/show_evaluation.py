import os
import pickle
import json
import cv2
import pandas as pd 

from keras_yolov2.utils import draw_boxes, draw_true_boxes,BoundBox



##### Parameters #####

# Path to evaluation history (either boxes or bad_boxes)
pickle_path = "/src/datafiles/yolo/pickles/histories/MobileNetV2-alpha=1.0_2023-04-21-09:33:55_0/random_50_files_in_test/boxes_MobileNetV2-alpha=1.0_random_50_files_in_test.p"

# Path to config filed use to evaluate
config_path = "/src/datafiles/yolo/configs/benchmark_configbees_detection_mobilenet_retrain_find_lr_lr_scheduler.json"

# Path to whole dataset
dataset_path = "/workspaces/projet_bees_detection_basile/folder/random_50_files_in_test.csv"

# Path to output folder
output_path = "/workspaces/projet_bees_detection_basile/folder/random_files_in_test/preds/"

##### Main #####

# Open pickle
with open(pickle_path, 'rb') as fp:
    img_boxes = pickle.load(fp)

# Open config file as a dict
with open(config_path) as config_buffer:
    config = json.load(config_buffer)

df_dataset=pd.read_csv(dataset_path,names=['filepath','xmin','ymin','xmax','ymax','label','width','height'])


# Make sure the output path exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Draw predicted boxes and save
for img in img_boxes:
    # Load image
    print(img)
    img_path = os.path.join(config["data"]["base_path"],img)
    print(img_path)
    frame = cv2.imread(img_path)

    # Get the true boxes
    true_boxes = df_dataset[df_dataset['filepath'] == os.path.join('BD_71',img)]
    true_boxes = true_boxes[['xmin', 'ymin', 'xmax', 'ymax','label']].values
    # Convert to bbox format
    true_boxes=[BoundBox(true_boxes[i][0],true_boxes[i][1],true_boxes[i][2],true_boxes[i][3],true_boxes[i][4]) for i in range(len(true_boxes))]


    # Draw true boxes
    frame = draw_true_boxes(frame, true_boxes, config['model']['labels'])

   # Draw predicted boxes
    frame = draw_boxes(frame, img_boxes[img], config['model']['labels'])
    
    # Save image
    cv2.imwrite(output_path + str.replace(img, '/', '_'), frame)
    