import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from keras_yolov2.preprocessing import parse_annotation_csv
from keras_yolov2.utils import (bbox_iou,
                                from_id_to_label_name,
                                compute_class_TP_FP_FN,
                                get_p_r_f1_global,
                                get_precision_recall_from_prediction_label,
                                results_metrics_per_classes)


########### INPUTS ############

#d'abord évaluer un fichier config
#mettre ce fichier config en hardcode sur ce fichier, le lancer et obtenir le graphique

# Path to evaluation history pickle file, starting with "boxes"
pickle_path = "/src/datafiles/yolo/pickles/histories/ResNet50_2023-04-18-16:14:01_0/test/boxes_ResNet50_test.p"

# Path to config file used to evaluate
config_path = "/src/datafiles/yolo/configs/benchmark_configbees_detection_resnet_cosine.json"



######## FUNCTIONS ###########

def load_k(k):
    """
    Loads the k-th image an its annotations
    :param k: index of the image
    :return: an array of annotations, each annotation is an array bbox-like (xmin, ymin, xmax, ymax, class)
    """
    annots = []

    # Loop on all objects
    for obj in images[k]['object']:
        annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], config['model']['labels'].index(obj['name'])]
        annots += [annot]

    # If there is no objects, create empty result
    if len(annots) == 0:
        annots = [[]]

    return np.array(annots)


# Open pickle
with open(pickle_path, 'rb') as fp:
    img_boxes = pickle.load(fp)


# Open config file as a dict
with open(config_path) as config_buffer:
    config = json.load(config_buffer)

# Load test CSV input file
images = []

test_csv_files = config['data']['test_csv_file']

if len(test_csv_files) == 0:
    raise ValueError('No test CSV file provided. Please provide at least one test CSV file.')

elif len(test_csv_files) == 1:
    images, _ = parse_annotation_csv(config['data']['test_csv_file'][0],
                                     config['model']['labels'],
                                     config['data']['base_path'])
                         
else:
    for test_csv_file in test_csv_files:
        images += parse_annotation_csv(test_csv_file,
                                        config['model']['labels'],
                                        config['data']['base_path'])[0]

# List labels
list_labels = config['model']['labels']

# Loop on
ious = np.linspace(0.0, 1.0, 10)
scores = np.linspace(0.0, 0.9, 10)

# Result in prf1 list
prf1 = []

it = 0
# Main loop
for iou_threshold in ious:
    res_iou = []
    for score_threshold in scores:
        predictions = []

        # iterate over all images in pickle file
        for k, img_name in enumerate(img_boxes): 
            pred_boxes = img_boxes[img_name]

            # Select boxes with high scores
            pred_boxes = [box.copy() for box in pred_boxes if box.get_score() > score_threshold]

            # Non max suppression
            for c in range(len(list_labels)):
                sorted_indices = list(reversed(np.argsort([box.classes[c] for box in pred_boxes])))

                for i in range(len(sorted_indices)):
                    index_i = sorted_indices[i]

                    if pred_boxes[index_i].classes[c] == 0:
                        continue
                    else:
                        for j in range(i + 1, len(sorted_indices)):
                            index_j = sorted_indices[j]

                            if bbox_iou(pred_boxes[index_i], pred_boxes[index_j]) >= iou_threshold:
                                pred_boxes[index_j].classes[c] = 0
            
            # Select boxes with high score now NMS is done
            pred_boxes = [box for box in pred_boxes if box.get_score() > score_threshold]
            
            # Extract boxes infos
            score = [box.score for box in pred_boxes]
            pred_labels = [box.get_label() for box in pred_boxes]

            # Store image infos
            labels_predicted = {}
            labels_predicted['img_name'] = img_name
            labels_predicted['predictions_id'] = pred_labels
            labels_predicted['predictions_name'] = from_id_to_label_name(list_labels, labels_predicted['predictions_id'])
            labels_predicted['score'] = score


            # Store true labels
            try :
                # Load annotations of the image
                annotation_k = load_k(k)

                # If there is no objects, create empty result
                if len(annotation_k[0]) == 0:
                    labels_predicted['true_id'] = 1
                    labels_predicted['true_name'] = ['unknown']
                else:
                    labels_predicted['true_id'] = list(annotation_k[:,4])
                    labels_predicted['true_name'] = from_id_to_label_name(list_labels, list(annotation_k[:,4]))
                
                # Compute TP FP FN TN
                compute_class_TP_FP_FN(labels_predicted)
                predictions.append(labels_predicted)
                

            except :
                Exception('Error while loading image : ' + img_name + '')


        # Compute global results
        class_metrics = get_precision_recall_from_prediction_label(predictions, list_labels)
        class_res = results_metrics_per_classes(class_metrics)
        p_global, r_global, f1_global = get_p_r_f1_global(class_metrics)

        # Compute mean results
        P_list = []
        R_list = []
        F1_list = []
        for res in class_res:
            if res['Specie'] in list_labels:
                P_list.append(res['Precision'])
                R_list.append(res['Rappel'])
                F1_list.append(res['F-score'])
        p_mean, r_mean, f1_mean =  np.mean(P_list), np.mean(R_list), np.mean(F1_list)
        
        # Add new results
        res_iou.append([p_global, r_global, f1_global, p_mean, r_mean, f1_mean])

        # Show progress
        it += 1
        progress = round(it / len(ious) / len(scores) * 100)
        print(f' {progress: 3}%' , end='\r')

    prf1.append(res_iou)

titles = [
    'Global precision',
    'Global recall',
    'Global f1-score',
    'Mean precision',
    'Mean recall',
    'Mean f1-score'
]

zlabels = ['Precision', 'Recall', 'F1-Score']

# Create scores and ious meshes
scores_mesh, ious_mesh = np.meshgrid(scores, ious)

# P-R-F1 As array
prf1 = np.array(prf1)

# save the plots in a folder
now=datetime.now()
output_path='/src/datafiles/yolo/outputs/plots_find_thresholds_on_{}'.format(now.strftime("%d-%m-%Y_%H:%M"))
os.makedirs(output_path)

# Plot figures
for i in range(len(titles)):

    # Plot figures one by one
    plt.figure(titles[i])
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(scores_mesh, ious_mesh, prf1[:, :, i], rstride=1, cstride=1, cmap='twilight_shifted', edgecolor='none')
    ax.set_title(titles[i])
    ax.set_xlabel('Score threshold')
    ax.set_ylabel('IoU threshold')
    ax.set_zlabel(zlabels[i % 3])
    plt.colorbar(surf)
    plt.savefig(output_path + '/' + titles[i] + '.png') 

# Find the best score and iou for each metric
best_score = {}
best_iou =  {}
for i,title in enumerate(titles):
    best_score[title] = scores[np.argmax(prf1[i % 3][:, i // 3])]
    best_iou[title] = ious[np.argmax(prf1[i % 3][:, i // 3])]

print('Best score and iou thresholds for each metric:')        
for title in titles:
    print(f' {title}: score = {best_score[title]}, iou = {best_iou[title]}')
