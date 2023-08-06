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
from keras_yolov2.frontend import YOLO
from keras_yolov2.utils import draw_boxes, list_images
from keras_yolov2.tracker import NMS, BoxTracker
from keras_yolov2.utils import compute_class_videos_TP_FP_FN
import time
import datetime

#on récupère les espèces du csv  

def obtain_list_especes_reelles(videos_path):
    with open("/src/datafiles/yolo/all_videos_annotated.csv", 'r') as f:
        reader = csv.reader(f)
        list_videos = list(reader)
    list_especes_reelles = []
    #on veut extraire le nom de la vidéo de videos_path
    video_name = videos_path.split("/")[-1]
    for video in list_videos:
        if video[0] == video_name:
            #on récupére tous les élements de la liste sauf le premier qui est le nom de la vidéo
            list_especes_reelles = video[1:]

    return list_especes_reelles

def predict_videos(videos_path, config_path, weights_path, list_especes_reelles):
    
    videos_format = ['.mp4', 'avi']
        # Load config file
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)


    # Create model
    yolo = YOLO(backend=config['model']['backend'],
                input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
                labels=config['model']['labels'],
                anchors=config['model']['anchors'],
                gray_mode=config['model']['gray_mode'])

    # Load weights
    yolo.load_weights(weights_path)

    # Tracker
    BT = BoxTracker()
    if os.path.splitext(videos_path)[1] in videos_format:
        file, ext = os.path.splitext(videos_path) #on enlève l'extension mp4

        # Chemin vers le dossier dans lequel on enregistre la vidéo

        save_path = "/home/acarlier/code/data_ornithoscope/birds_videos/predicted/" + file.split("/")[-1] 

        video_out = '{}_detected.avi'.format(save_path)

    
    # On étudie la vidéo

    video_reader = cv2.VideoCapture(videos_path)


    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)) # .get(cv2.CAP_PROP_FRAME_COUNT)) sert à récupérer le nombre de frames dans la vidéo
    

    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    video_writer = cv2.VideoWriter(video_out,
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    15.0,
                                    (frame_w, frame_h))

    
    for _ in tqdm(range(nb_frames)):
    
        # Read video
        ret, frame = video_reader.read()
        if not ret:
            running = False
            continue

        # Predict
        boxes = yolo.predict(frame,
                                iou_threshold=config['valid']['iou_threshold'],
                                score_threshold=config['valid']['score_threshold'])

        # Decode and draw boxes
        boxes = NMS(boxes)
        boxes = BT.update(boxes).values() #BT est égale au box tracker défini dans tracker.py
        
        frame = draw_boxes(frame, boxes, config['model']['labels'])

        # Write video output
        video_writer.write(np.uint8(frame))

    video_reader.release()
    video_writer.release()
    
    # History infos
    
    res = {}
    for id, history in zip(BT.tracker_history.keys(), BT.tracker_history.values()):
        res[id] = {}
        for class_id, trust in history:
            class_label = config['model']['labels'][class_id]
            if class_label in res[id]:
                if trust > 0.5:
                    res[id][class_label] += trust
                else:
                    res[id][class_label] = trust
            else:
                    res[id][class_label] = trust
                
                
    
    print("resultat sans treeshold sur la valeur trust de chaque espèce:",res) #dictionnaire avec l'espèce prédite et un score de confiance en respectant les ordres de passage.

    #on veut connaitre le nombre de frames où l'espèce est prédite avec une confiance supérieure à 0.9 pour chaque id
    
    res2 = {}
    for id, history in zip(BT.tracker_history.keys(), BT.tracker_history.values()):
        res2[id] = {}
        for class_id, trust in history:
            class_label = config['model']['labels'][class_id]
            if class_label in res2[id]:
                if trust > 0.5: #ne bouge pas
                    res2[id][class_label] += 1
                else:
                    res2[id][class_label] = 1
            else:
                    res2[id][class_label] = 1
                    
    print("Nombre de frames avec treeshold sur la valeur trust de chaque espèce:",res2) #dictionnaire avec l'espèce prédite et un score de confiance en respectant les ordres de passage.               
    
    #On ne veut garder seulement les clés de res qui ont une valeur supérieure à 20

    #cas n°1: une espèce peut en remplacer une autre sur la mangeoire et on prend toutes les espèces d'une même id ayant un trust supérieur à 20
    
    # list_especes_predites = []
    # id_res = list(res.keys())
    # for id in id_res:
    #     list_especes = list(res[id].keys())
    #     removed = []
    #     for especes in list_especes:
            
    #         if res[id][especes] < 20:
    #             print("espece supprimée:",especes)
    #             removed.append(especes)
    #             print("nouvelle liste des espèces:",list_especes) 


    #     for espece in removed: #méthode de code classique! Ne jamais bouclé sur une liste dont on supprime des éléments, même si on n'utilise pas ses indices
    #         list_especes.remove(espece)

    #     if len(list_especes) > 0: #cas n°1 où on suppose qu'une espèce peut en remplacer une autre en la poussant et alors garder la même bbox: on prend toutes les espèces d'une même id et non le trust max d'une même id
    #         for i in range(len(list_especes)):
                # list_especes_predites.append(list_especes[i])

    #cas n°2: une espèce peut en remplacer une autre sur la mangeoire et on prend la plus grande bbox de chaque id
    
    list_especes_predites = []
    id_res = list(res.keys())
    for id in id_res:
        list_especes = list(res[id].keys())
        #on ne veut garder que la clé de res[id] qui a la plus grande valeur
        max_trust = max(list(res[id].values()))
        list_especes_predites.append(list(res[id].keys())[list(res[id].values()).index(max_trust)])
    print("liste des espèces prédites:",list_especes_predites)
       
   #cas n°3: on prend le max des trust si juste une seul id est supérieur à 50, sinon on prend les deux espèces de la même id avec un trust supérieur à 50
   
    print("liste des espèces finales",list_especes_predites)
    print("\n")

    #on veut écrire le nombre de frame pour chaque espèce apparaissant dans la vidéo, mais seulement pour les vidéos bien prédites
    
    #if set(list_especes_predites) == set(list_especes_reelles):
        
    #d'abord on cherche la durée de la videos en secondes
    
    # create video capture object
    data = cv2.VideoCapture(videos_path)
    
    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(data.get(cv2.CAP_PROP_FPS))
    
    # calculate dusration of the video
    seconds = int(frames / fps)
    video_time = str(datetime.timedelta(seconds=seconds))
        

    with open("/src/datafiles/yolo/count_frame.csv", "a", newline='') as f:
        writer = csv.writer(f)
        L = [videos_path]
        for index in res2:  
            for especes in res2[index]:
                if especes in list_especes_predites:
                    if res2[index][especes] > 7: #si l'apparition est supérieur à 7 frames (soit environ un quart de seconde minimal, on considère que c'est le temps minimal raisonnable de détection pour que l'espèce soit présente sur la mangeoire)
                        
                        time_espece = res2[index][especes]/fps
                        # writer.writerow([time_espece])
                        L.append(time_espece)
        writer.writerow(L) #onveut tout afficheer sur la même ligne du csv donc on utilise une liste
    return list_especes_predites


#on calcules metrics image par image

def compute_TP_FP_FN(list_especes_predites, list_especes_reelles):
    TP = 0
    FP = 0
    FN = 0
    for especes in list_especes_predites:
        if especes in list_especes_reelles:
            TP += 1
        else:
            FP += 1
    for especes in list_especes_reelles:
        if especes not in list_especes_predites:
            FN += 1
    return TP, FP, FN  #on retourne les valeurs de TP, FP et FN, qui sont des listes, pour une image

#on affiche les espèces qui sont TP, FP et FN

def display_TP_FP_FN(list_especes_predites, list_especes_reelles):
    TP_list = []
    FP_list = []
    FN_list = []
    for especes in list_especes_predites:
        if especes in list_especes_reelles:
            TP_list.append(especes)
        else:
            FP_list.append(especes)
    for especes in list_especes_reelles:
        if especes not in list_especes_predites:
            FN_list.append(especes)
    return TP_list, FP_list, FN_list  #on retourne les valeurs de TP, FP et FN, qui sont des listes, pour une image
#on calcule les metrics globales pour un ensemble de vidéos

def compute_F1_score_for_videos(list_especes_predites, list_especes_reelles):
    TP, FP, FN = compute_TP_FP_FN(list_especes_predites, list_especes_reelles)
    if TP+FP != 0:
        precision = TP/(TP+FP)
    else:
        precision = 0
    if TP+FN != 0:
        rappel = TP/(TP+FN)
    else:
        rappel = 0
    if precision+rappel != 0:
        F1_score = 2*precision*rappel/(precision+rappel)
    else:
        F1_score = 0

    return [precision, rappel, F1_score]


#on veut générer une liste de vidéos à tester pour ensuite lancer evaluate_on_videos sur chacune d'entre elles et obtenir tous les F1 score, rappel, précision

def generate_list_videos_to_test(): #à changer lorsque l'on obtiendra de nouvelles vidéos
    with open("/src/datafiles/yolo/all_videos.csv", 'r') as f:
        reader = csv.reader(f)
        list_videos = list(reader)
        #on transforme la liste de snoms de vidéos en liste de chemins vers les vidéos, en prenant en compte les sous-dossiers selon les tasks
        base_path = "/home/acarlier/code/data_ornithoscope/birds_videos/2022"
        list_of_tasks = ['balacet', 'C1', 'C4', 'gajan']
        list_paths_videos = []
        for tasks in list_of_tasks:
            for videos in list_videos:
                if tasks in videos[0]:
                    videos = base_path+'/'+tasks+'/'+videos[0]
                    list_paths_videos.append(videos)
    return list_paths_videos

        

#on veut écrire une fonction qui affiche les résultats moyens des F1 score, rappel, précision par class

list_of_classes = ["Anthophila"]
class_metrics = []
for classes in list_of_classes:
        class_metrics.append({'Specie':classes,'TP':0, 'FP':0, 'FN':0})

def get_precision_recall_from_prediction_label(TP_list, FP_list, FN_list ):
    #print(f'TP {TP}, FN {FN}, FP {FP}')
    
    for lab in TP_list:
        class_metrics[list_of_classes.index(lab)]['TP'] += 1

    for lab in FN_list:
        class_metrics[list_of_classes.index(lab)]['FN'] += 1
    
    #print(class_metrics) renvoie une liste de dictionnaires contenant les métriques de chaque classe

    
    for lab in FP_list: #FP quel type: liste ou string?
        class_metrics[list_of_classes.index(lab)]['FP'] += 1
    
    return class_metrics

def results_metrics_per_classes(class_metrics):
    class_res = []  
    i = 0
    for class_bird in class_metrics:
        specie = class_bird['Specie']
        TP = class_bird['TP']
        FP = class_bird['FP']
        FN = class_bird['FN']
        #print('TP-FP-FN')
        #print(f'{specie} {TP} {FP} {FN}')
        if (TP+FP) == 0: 
            P = 0
        else: 
            P = TP/(TP+FP)
        if (TP+FN) == 0: 
            R = 0
        else:
            R = TP/(TP+FN)
        if P == 0 and R == 0:
            F_score = 0
        else:
            F_score = 2*P*R/(P+R)
        #print('P-R-F_score')
        #print(f'{specie} {P} {R} {F_score}')
        class_res.append({'Specie': specie,'Precision': round(P,3), 'Rappel': round(R,3), 'F-score': round(F_score,3)})
        i += 1
    return class_res



def print_results_metrics_per_classes(class_res):
    seen_valid = ["Anthophila"]
    P_list = []
    R_list = []
    F1_list = []
    for res in class_res:
        if res['Specie'] in seen_valid:
            P = res['Precision']
            R = res['Rappel']
            F1 = res['F-score']
            P_list.append(P)
            R_list.append(R)
            F1_list.append(F1)
            print(f"Specie = {res['Specie']}, Precision = {P} - Rappel = {R} - F-score = {F1} ")
    return round(np.mean(P_list), 3), round(np.mean(R_list), 3), round(np.mean(F1_list), 3)

#fonction pour détecter les images mal annotées

def detect_bad_annotated_images_and_save_path(list_especes_predites, list_especes_reelles, videos_path):
    with open('/src/datafiles/yolo/comparate_results_videos/results_prediction.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([videos_path, list_especes_predites, list_especes_reelles])
        print(f' register video {videos_path} done')
    #si on ne retrouve pas les même espèces dans les deux liste alors on enregistre le nom de la vidéo dans un fichier csv
    if set(list_especes_predites) != set(list_especes_reelles):
        with open('/src/datafiles/yolo/videos_with_diff_species_max20.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([videos_path, list_especes_predites, list_especes_reelles])
            
            

