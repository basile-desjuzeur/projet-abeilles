import json
import cv2
import time
import datetime
import argparse
import os
import csv
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from keras_yolov2.utils import enable_memory_growth,print_results_metrics_per_classes, print_ecart_type_F1
from keras_yolov2.utils_for_evaluate_videos import predict_videos, obtain_list_especes_reelles,compute_F1_score_for_videos,generate_list_videos_to_test,compute_TP_FP_FN,get_precision_recall_from_prediction_label,results_metrics_per_classes,display_TP_FP_FN,detect_bad_annotated_images_and_save_path
from keras_yolov2.frontend import YOLO
from keras_yolov2.map_evaluation import MapEvaluation
import argparse
import json
import os
import pickle
from datetime import datetime
import tensorflow as tf


argparser = argparse.ArgumentParser(
  description='Predict tflite model in real time or with movies / images')

argparser.add_argument(
  '-c',
  '--conf',
  default='/src/datafiles/yolo/configs/benchmark_configbees_detection.json',
  type=str,
  help='path to configuration file')

argparser.add_argument(
  '-w',
  '--weights',
  #default="/workspaces/projet_bees_detection/bees_detection/src/datafiles/yolo/saved_weights/best_model_bestLoss.h5",
  type=str,
  help='path to pretrained weights')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    #videos_path = args.input #chemin vers une seule vidéo pour commencer l'evaluate
    

    enable_memory_growth()

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    if weights_path == '':
        weights_path = config['train']['pretrained_weights']
        
    #on récupère la liste des viéos à tester
    list_videos_to_test = generate_list_videos_to_test()

    #on veut enlever les oiseaux non dans le train
    
    list_non_train = []
    remove = []
    for video in list_videos_to_test:
        for especes in list_non_train:
            if especes in video:
                remove.append(video)
    for video in remove:
        list_videos_to_test.remove(video)

    #on prend les prédictions du modèle
    dic_res_brut = {'precision':[], 'rappel': [], 'F1 score':[]}

    for videos_path in list_videos_to_test: 
        
        print(videos_path)
        list_especes_reelles = obtain_list_especes_reelles(videos_path)
        list_especes_predites = predict_videos(videos_path, config_path, weights_path, list_especes_reelles)

        detect_bad_annotated_images_and_save_path(list_especes_predites, list_especes_reelles, videos_path)

        res = compute_F1_score_for_videos(list_especes_predites, list_especes_reelles) #renvoie une liste [precision, rappel, F1_score] pour chaque image
        TP, FP, FN = display_TP_FP_FN(list_especes_predites, list_especes_reelles)
        class_metrics = get_precision_recall_from_prediction_label(TP, FP, FN )
        class_res = results_metrics_per_classes(class_metrics)

        dic_res_brut['precision'].append(res[0])
        dic_res_brut['rappel'].append(res[1])
        dic_res_brut['F1 score'].append(res[2])
       
    #on calcule les moyennes pour chaque mesure
    precision_moyenne = np.mean(dic_res_brut['precision'])
    rappel_moyen = np.mean(dic_res_brut['rappel'])
    F1_score_moyen = np.mean(dic_res_brut['F1 score'])
    print("Métriques obtenues par classe:")
    print("\n")
    seen_valid = ["Anthophila"]
    print(print_results_metrics_per_classes(class_res, seen_valid))
    print("-----------------------")
    print("Moyennes des mesures pour les vidéos testées:")
    print("\n")
    #on veut afficher les résultats avec trois chiffres après la virgule
    print('precision moyenne : ', round(precision_moyenne,3))
    print('rappel moyen : ', round(rappel_moyen,3))
    print('F1 score moyen : ', round(F1_score_moyen,3))

if __name__ == '__main__':
    _args = argparser.parse_args()
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    with tf.device('/GPU:' + gpu_id):
        _main_(_args)



