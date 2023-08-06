#! /usr/bin/env python3
import argparse
import json
import os
import pickle
from datetime import datetime
import tensorflow as tf

from keras_yolov2.preprocessing import parse_annotation_csv
from keras_yolov2.preprocessing import BatchGenerator
from keras_yolov2.utils import enable_memory_growth, print_results_metrics_per_classes, print_ecart_type_F1
from keras_yolov2.frontend import YOLO
from keras_yolov2.map_evaluation import MapEvaluation

# tf.debugging.set_log_device_placement(True)

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='/src/datafiles/yolo/configs/benchmark_configfinal_config.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='/src/datafiles/yolo/saved_weights/Best_model_bestLoss.h5',
    help='path to pretrained weights')

argparser.add_argument(
    '-l',
    '--lite',
    default='',
    type=str,
    help='Path to tflite model')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    lite_path = args.lite

    enable_memory_growth()

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if weights_path == '':
        weights_path = config['train']['pretrained_weights']

    ##########################
    #   Parse the annotations
    ##########################
    without_valid_imgs = False

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation_csv(config['data']['train_csv_file'],
                                                    config['model']['labels'],
                                                    config['data']['base_path'])

    # remove samples without objects in the image
    for i in range(len(train_imgs)-1, 0, -1):
        if len(train_imgs[i]['object']) == 0:
            del train_imgs[i]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(
            set(train_labels.keys()))

        if len(overlap_labels) < len(config['model']['labels']):
            print(
                'Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Evaluate on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        with open("labels.json", 'w') as outfile:
            json.dump({"labels": list(train_labels.keys())}, outfile)

    ########################
    #   Construct the model
    ########################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=(config['model']['input_size_h'],
                            config['model']['input_size_w']),
                labels=config['model']['labels'],
                anchors=config['model']['anchors'],
                gray_mode=config['model']['gray_mode'],
                freeze=config['train']['freeze'])

    #########################################
    #   Load the pretrained weights (if any)
    #########################################


    if weights_path != '':
        print("Loading pre-trained weights in", weights_path)
        yolo.load_weights(weights_path) 
    elif os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in",
              config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])
    else:
        raise Exception("No pretrained weights found.")

    # Use tflite
    if lite_path != '':

        yolo.load_lite(lite_path)

    #########################
    #   Evaluate the network
    #########################

    test_csv_files = config['data']['test_csv_file']
    directory_name = f"{config['model']['backend']}_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

    parent_dir = config['data']['saved_pickles_path']
    path = os.path.join(parent_dir, directory_name)
    print("Directory name for metrics: ", path)
    count = 0

    # checks if the directory were the pickles will be saved exists
    while True:
        try:
            os.mkdir(path + f'_{count}')
            break
        except:
            count += 1

    path += f'_{count}'

    if type(test_csv_files) is str:
        test_csv_files = [test_csv_files]

    for test_csv_file in test_csv_files:

        if os.path.exists(test_csv_file):
            print(f"\n \nParsing {test_csv_file.split('/')[-1]}\n \n")

            test_imgs, seen_valid_labels = parse_annotation_csv(test_csv_file,
                                                                config['model']['labels'],
                                                                config['data']['base_path'])

            generator_config = {
                'IMAGE_H': yolo._input_size[0],
                'IMAGE_W': yolo._input_size[1],
                'IMAGE_C': yolo._input_size[2],
                'GRID_H': yolo._grid_h,
                'GRID_W': yolo._grid_w,
                'BOX': yolo._nb_box,
                'LABELS': yolo.labels,
                'CLASS': len(yolo.labels),
                'ANCHORS': yolo._anchors,
                'BATCH_SIZE': 4,
                'TRUE_BOX_BUFFER': 10
            }

            test_generator = BatchGenerator(test_imgs,
                                            generator_config,
                                            norm=yolo._feature_extractor.normalize,
                                            jitter=False,
                                            shuffle=False)
            
            test_eval = MapEvaluation(yolo, test_generator,
                                      iou_threshold=config['valid']['iou_threshold'],
                                      score_threshold=config['valid']['score_threshold'],
                                      label_names=config['model']['labels'],
                                      model_name=config['model']['backend'])

            print('\nNumber of valid images: \n', len(test_imgs))

            print('\nComputing metrics per classes...\n')

            (boxes_preds, bad_boxes_preds,
             class_predictions, class_metrics, class_res, class_p_global, class_r_global, class_f1_global,
             bbox_predictions, bbox_metrics, bbox_res, bbox_p_global, bbox_r_global, bbox_f1_global,
             ious_global, intersections_global
             ) = test_eval.compute_P_R_F1()
            
            print('\nMetrics computed\n')

            test_name = test_csv_file.split('/')[-1].split('.')[0]

            print("For", test_name)
            print('VALIDATION LABELS: ', seen_valid_labels)

            print ('-'*70)
            print('\033[1m'+'\n\tFinal results:')


            print('\nClass metrics:')
            class_mean_P, class_mean_R, class_mean_F1 = print_results_metrics_per_classes(
                class_res, seen_valid_labels)
            print(
                f"Class globals: P={class_p_global} R={class_r_global} F1={class_f1_global}")
            print(
                f"Class means: P={class_mean_P} R={class_mean_R} F1={class_mean_F1}")

            print('\nBBox metrics:')
            bbox_mean_P, bbox_mean_R, bbox_mean_F1 = print_results_metrics_per_classes(
                bbox_res, seen_valid_labels)
            print(
                f"BBox globals: P={bbox_p_global} R={bbox_r_global} F1={bbox_f1_global}")
            print(
                f"BBox means: P={bbox_mean_P} R={bbox_mean_R} F1={bbox_mean_F1}")

            print(f"Overall IoU on true positives: {round(ious_global,3)}")
            print(
                f"Proportion of true box covered by pred box on true positives: {round(intersections_global,3)}")
            print ('-'*70)

            # Save the results in a folder with the name of the csv file

            new_path = os.path.join(path, test_name)
            os.mkdir(new_path)

            global_results = [class_p_global, class_r_global, class_f1_global]
            pickle.dump(class_predictions, open(
                f"{new_path}/prediction_TP_FP_FN_{config['model']['backend']}_{test_name}.p", "wb"))
            pickle.dump(class_metrics, open(
                f"{new_path}/TP_FP_FN_{config['model']['backend']}_{test_name}.p", "wb"))
            pickle.dump(class_res, open(
                f"{new_path}/P_R_F1_{config['model']['backend']}_{test_name}.p", "wb"))
            pickle.dump(global_results, open(
                f"{new_path}/P_R_F1_global_{config['model']['backend']}_{test_name}.p", "wb"))
            pickle.dump(boxes_preds, open(
                f"{new_path}/boxes_{config['model']['backend']}_{test_name}.p", "wb"))
            pickle.dump(bad_boxes_preds, open(
                f"{new_path}/bad_boxes_{config['model']['backend']}_{test_name}.p", "wb"))

    return class_p_global, class_r_global, class_f1_global, bbox_p_global, bbox_r_global, bbox_f1_global, round(ious_global, 3), round(intersections_global, 3)


if __name__ == '__main__':
    _args = argparser.parse_args()
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    with tf.device('/GPU:' + gpu_id):
        _main_(_args)

