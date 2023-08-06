import json
import cv2
import time
import datetime
from datetime import datetime
import argparse
import os
import csv
from tqdm import tqdm
import pandas as pd

import numpy as np
import tensorflow as tf

from keras_yolov2.frontend import YOLO
from keras_yolov2.utils import draw_boxes, list_images, bbox_iou
from keras_yolov2.tracker import NMS, BoxTracker


argparser = argparse.ArgumentParser(
  description='Predict tflite model in real time or with movies / images')

argparser.add_argument(
  '-c',
  '--conf',
  default="/datafiles/yolo/configs/final_config.json",
  type=str,
  help='path to configuration file')

argparser.add_argument(
  '-w',
  '--weights',
  default='/datafiles/yolo/saved_weights/Best_model_bestLoss.h5',
  type=str,
  help='path to pretrained weights')

argparser.add_argument(
  '-l',
  '--lite',
  default='',
  type=str,
  help='Path to tflite model')

argparser.add_argument(
  '-r',
  '--real_time',
  default=False,
  type=bool,
  help='use a camera for real time prediction')

argparser.add_argument(
  '-i',
  '--input',
  type=str,
  default='',
  help='path to an image,a csv, a video or a folder of images')

argparser.add_argument(
  '-o',
  '--output',
  type=str,
  default='csv_input',  # output in /datafiles/crop/predict_csv/
  help='Output format, img (default) or csv / csv_input') # set to csv_input


def _main_(args):
  config_path = args.conf
  weights_path = args.weights
  image_path = args.input
  use_camera = args.real_time
  lite_path = args.lite
  output_format = args.output
  input = args.input

  videos_format = ['.mp4', 'avi']

  # Load config file
  with open(config_path) as config_buffer:
    config = json.load(config_buffer)

  # Set weights path
  if weights_path == '':
    weights_path = config['train']['pretrained_weights']

  # Create model
  yolo = YOLO(backend=config['model']['backend'],
              input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
              labels=config['model']['labels'],
              anchors=config['model']['anchors'],
              gray_mode=config['model']['gray_mode'])

  # Load weights
  yolo.load_weights(weights_path)

  # Use tflite
  if lite_path != '':
    yolo.load_lite(lite_path)

  # Tracker
  BT = BoxTracker()

  ### Real Time
  if use_camera:

    # Set video capture
    video_reader = cv2.VideoCapture(int(image_path))
    
    # Variables to calculate FPS
    start_time = time.time()
    counter, fps = 0, 0
    fps_avg_frame_count = 10

    # Main loop
    running = True
    while running:
      counter += 1

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
      boxes = BT.update(boxes).values()
      
      # Draw boxes
      frame = draw_boxes(frame, boxes, config['model']['labels'])

      # Show the date
      current_time = str(datetime.datetime.utcfromtimestamp(int(time.time())))
      cv2.putText(frame, current_time, (20, 20), cv2.FONT_HERSHEY_PLAIN,
          1, (0, 255, 0), 2)

      # Calculate the FPS
      if counter % fps_avg_frame_count == 0:
          end_time = time.time()
          fps = fps_avg_frame_count / (end_time - start_time)
          start_time = time.time()
      
      # Show the FPS
      fps_text = '{:02.1f} fps'.format(fps)
      cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_PLAIN,
          1, (0, 255, 0), 2)

      # Show frame
      cv2.imshow("frame", frame)

      # Quit
      key = cv2.waitKey(1)
      if key == ord("q") or key == 27:
        running = False
  
  ### Video
  elif os.path.splitext(image_path)[1] in videos_format:
    file, ext = os.path.splitext(image_path) #on enlève l'extension mp4

    # Chemin vers le dossier dans lequel on enregistre la vidéo

    save_path = "/home/acarlier/code/data_ornithoscope/birds_videos/predicted/" + file.split("/")[-1] 

    video_out = '{}_detected.avi'.format(save_path)

  
  # On étudie la vidéo

    video_reader = cv2.VideoCapture(image_path)

  
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
      
      print(boxes)

      frame = draw_boxes(frame, boxes, config['model']['labels'])

      # calcul du iou score avec la fonction iou_score

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
          res[id][class_label] += trust
        else:
          res[id][class_label] = trust
      
    print("resultat sans treeshold sur la valeur trust de chaque espèce:",res) #dictionnaire avec l'espèce prédite et un score de confiance en respectant les ordres de passage.
    
    #On ne veut garder seulement les clés de res qui ont une valeur supérieure à 20

    res2 = res
    for id in res:
      print("id:",id)
      list_especes = list(res[id].keys())
      for especes in list_especes:
        if res[id][especes] < 20:
          list_especes.pop(list_especes.index(especes))

    res3 = {}
    for id in res2:
      res3[id] = {}
      for especes in list_especes:
        res3[id][especes] = res2[id][especes]

    print(list_especes)
    print("resultat final:",res3)

  ###  CSV
  # TODO: add csv input
  elif str(input).endswith('.csv'):

    print('-'*30)
    print("Predicting from csv file")
    print('-'*30)

    # read csv
    df = pd.read_csv(input)

    # get paths
    paths = df['path']

    # Create output csv
    if output_format == 'csv_input':
      
      # get output csv name
      now=datetime.now()
      csv_name = input.split('/')[-1].split('.')[0]
      ouptut_name = 'predictions_{}_{}.csv'.format(csv_name, now.strftime("%Y-%m-%d_%H-%M"))
      detected_csv = os.path.join('/datafiles/crop/predict_csv/', ouptut_name)
      print("Predictions will be saved in {}".format(detected_csv))

      # create csv
      f = open(detected_csv, 'w')
      writer = csv.writer(f)
      


    # list to store errors
    errors = []
    count_errors = 0


    # predict
    for path in tqdm(paths):
        
        # Open image
        frame = cv2.imread(path)

        try:
          boxes = yolo.predict(frame,
                              iou_threshold=config['valid']['iou_threshold'],
                              score_threshold=config['valid']['score_threshold'])
        
        except:
          print("Error with image {}".format(path))
          count_errors+=1
          errors.append(path)
          continue 

         # good for image classif    ## TODO: add other output formats
        if output_format == 'csv_input':
          image_h, image_w, _ = frame.shape
          for box in boxes:
            row = [
                path,
                int(box.xmin * image_w),
                int(box.ymin * image_h),
                int(box.xmax * image_w),
                int(box.ymax * image_h),
                config['model']['labels'][box.get_label()],
                image_w,
                image_h
              ]
            writer.writerow(row)

    # close csv
    f.close()



  ### Image
  else:
    count_errors=0
    errors=[]

    # One image
    if os.path.isfile(image_path):

      # Open image
      frame = cv2.imread(image_path)

      # Predict
      try:
        boxes = yolo.predict(frame,
                              iou_threshold=config['valid']['iou_threshold'],
                              score_threshold=config['valid']['score_threshold'])
        
      except:
        count_errors+=1
        errors.append(image_path)
        print("Error happened with ",image_path)
        return 

      #Avec bbox_iou on veut calculer le iou score de toutes les bounding boxes prédites et les afficher
      for i in range(len(boxes)):
        for j in range(len(boxes)):
          if i != j:
            print("iou score:",bbox_iou(boxes[i],boxes[j]))


      # Draw boxes
      frame = draw_boxes(frame, boxes, config['model']['labels'])
      # Write image output                                                      # no csv for single file 
      cv2.imwrite(image_path[:-4] + '_lite_detected' + image_path[-4:], frame)  ## problem for jpeg vs jpg ... os.path.splitext
  


    # Image folder
    else:
      if output_format == 'img':

        # Create output image folder
        detected_images_path = os.path.join(image_path, "detected_images")

        if not os.path.exists(detected_images_path):
          os.mkdir(detected_images_path) 

      elif output_format.startswith('csv'):

        # Create output csv
        # Create output image file/folder as input f
        now=datetime.now()
        path='/datafiles/crop/predict_csv/'
        detected_csv = os.path.join(path, "detected_images_{}.csv".format(now.strftime("%Y-%m-%d_%H-%M")))

        print("Predictions will be saved in {}".format(detected_csv))

        f = open(detected_csv, 'w')
        writer = csv.writer(f)

      images = list(list_images(image_path))


      if len(images) == 0:
        raise Exception("No images found in {}, it may be due to the fact that images have no extension in their file name, if so, run /datafiles/yolo/inputs/put_extension.py".format(image_path))
      
      for fname in tqdm(images):
        # Open image
        frame = cv2.imread(fname)
        
        # Predict

        try:
          boxes = yolo.predict(frame,
                              iou_threshold=config['valid']['iou_threshold'],
                              score_threshold=config['valid']['score_threshold'])
        
        except:
          print("Error with image {}".format(fname))
          count_errors+=1
          errors.append(fname)
          continue 

        if output_format == 'img':
          image = draw_boxes(frame, boxes, config['model']['labels'])
          fname = os.path.basename(fname)
          cv2.imwrite(os.path.join(image_path, "detected", fname), image)


        # sum up the detection
        elif output_format == 'csv':
          for box in boxes:
            row = [fname, box.xmin, box.ymin, box.xmax, box.ymax, box.score,box.get_label()]
            writer.writerow(row)

        # good for image classif    
        elif output_format == 'csv_input':
          image_h, image_w, _ = frame.shape
          for box in boxes:
            row = [
                fname,
                int(box.xmin * image_w),
                int(box.ymin * image_h),
                int(box.xmax * image_w),
                int(box.ymax * image_h),
                config['model']['labels'][box.get_label()],
                image_w,
                image_h
              ]
            writer.writerow(row)
              
      if output_format.startswith('csv'):
        f.close()

      print("Predictions saved in {}".format(detected_csv))

  # Errors handling
  if count_errors > 0:
    print("Errors occured on {} pictures".format(count_errors))

    # store the errors in a text file
    with open(os.path.join(image_path, "errors.txt"), "w") as f:
      for error in errors:
        f.write(error) 

    print("Exhaustive count of errors stored in {}".format(os.path.join(image_path, "errors.txt")))

  

if __name__ == '__main__':
  _args = argparser.parse_args()
  gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
  with tf.device('/GPU:' + gpu_id):
    _main_(_args)