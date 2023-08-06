from tqdm import tqdm

from tensorflow.keras.callbacks import Callback, TensorBoard
import numpy as np 
from tensorflow.python.ops import summary_ops_v2
from keras_yolov2.utils import compute_overlap, compute_ap

from .utils import (from_id_to_label_name,
                    compute_class_TP_FP_FN,
                    results_metrics_per_classes,
                    get_precision_recall_from_prediction_label,
                    get_precision_recall_from_prediction_box,
                    get_p_r_f1_global,
                    compute_bbox_TP_FP_FN,
                    BoundBox)


class MapEvaluation(Callback):
    """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
    """

    def __init__(self, yolo, generator,
                 iou_threshold=0.5,
                 score_threshold=0.5,
                 save_path=None,
                 period=1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None,
                 label_names=[],
                 model_name=''):

        super().__init__()
        self._yolo = yolo
        self._generator = generator
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold
        self._save_path = save_path
        self._period = period
        self._save_best = save_best
        self._save_name = save_name
        self._tensorboard = tensorboard
        self._label_names = label_names
        self._model_name = model_name

        self.bestMap = 0

        if not isinstance(self._tensorboard, TensorBoard) and self._tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")


    def compute_P_R_F1(self):

        """
        Compute Precision, Recall and F1-Score.
        """

        # Lists TP, FP and FN per image as a list of dicts
        class_predictions, bbox_predictions = [], []

        # Lists predict boxes per image
        boxes_preds, bad_boxes_preds = {}, {}

        # Lists ious and intersections per image
        ious_global = []
        intersections_global = []

        # Loop on every image of the dataset to evaluate
        for i in tqdm(range(self._generator.size())):


            # Predict the image
            image, img_name = self._generator.load_image(i)
            img_h, img_w = image.shape[0:2]
            pred_boxes = self._yolo.predict(image,
                                            iou_threshold=self._iou_threshold,
                                            score_threshold=self._score_threshold)
            
  
            # Load true results of the image as an array of BoundBox like lines
            annotation_i = self._generator.load_annotation(i) 

            # Convert annotations to BoundBoxes
            if annotation_i != []:
                true_boxes = [
                    BoundBox(
                        box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h, 1,
                        [1 if c == box[4] else 0 for c in range(len(self._label_names))]
                    ) for box in annotation_i
                ]     
            else:
                true_boxes = []
            
            #########
            # Bbox metrics
            #########

            # Compute and add TP, FP and FN to the bbox prediction list
            bbox_preddicted, ious_img , intersections_img = compute_bbox_TP_FP_FN(pred_boxes, true_boxes, self._label_names,iou_threshold=self._iou_threshold)
            bbox_predictions.append(bbox_preddicted)
            ious_global.extend(ious_img)
            intersections_global.extend(intersections_img)

            #########
            # Class metrics
            #########


            # Create class predicted dict
            class_preddicted = {} 
            class_preddicted['img_name'] = img_name
            class_preddicted['predictions_id'] = [box.get_label() for box in pred_boxes] #exctraction des id de la liste des boxes prédites
            class_preddicted['predictions_name'] = from_id_to_label_name(self._label_names, class_preddicted['predictions_id']) #renvoie le nom des prédictions
            class_preddicted['score'] = [box.score for box in pred_boxes]
            if len(annotation_i[0]) == 0:
                class_preddicted['true_id'] = 0
                class_preddicted['true_name'] = ['EMPTY']
            else:
                class_preddicted['true_id'] = list(annotation_i[:,4]) #exctraction des id de la liste des boxes prédites
                class_preddicted['true_name'] = from_id_to_label_name(self._label_names, list(annotation_i[:,4]))#renvoie le nom qui aurait dû être prédit
            
            
            # Compute and add TP, FP and FN to the class prediction list
            #A partir de cettel ligne on rajoute TP, FP et FN à class_preddicted
            
            compute_class_TP_FP_FN(class_preddicted)  
            class_predictions.append(class_preddicted)

            # Store predicted bounding box in 
            boxes_preds[img_name] = pred_boxes
            if (len(class_preddicted['FP'] + class_preddicted['FN'] + bbox_preddicted['FP'] + bbox_preddicted['FN']) > 0):
                bad_boxes_preds[img_name] = pred_boxes
        

        # Compute P, R and F1 with the class metrics
        class_metrics = get_precision_recall_from_prediction_label(class_predictions, self._label_names) #les labels_name sont ici les labels du backend (voir comment il appelle map-evaluate dans evaluate.py)
        class_res = results_metrics_per_classes(class_metrics)
        class_p_global, class_r_global, class_f1_global = get_p_r_f1_global(class_metrics)


        # Compute P, R and F1 with the bbox metrics
        bbox_metrics = get_precision_recall_from_prediction_box(bbox_predictions, self._label_names)
        bbox_res = results_metrics_per_classes(bbox_metrics)
        bbox_p_global, bbox_r_global, bbox_f1_global = get_p_r_f1_global(bbox_metrics)


        # Compute IoU and intersection for true positives
        if len(ious_global) == 0:
            ious_global = 0 
        else:
            ious_global = sum(ious_global)/len(ious_global)

        if len(intersections_global) == 0:
            intersections_global =0

        else:
            intersections_global = sum(intersections_global)/len(intersections_global)
        

        return (boxes_preds, bad_boxes_preds,
                class_predictions, class_metrics, class_res, class_p_global, class_r_global, class_f1_global,
                bbox_predictions, bbox_metrics, bbox_res, bbox_p_global, bbox_r_global, bbox_f1_global,
                ious_global, intersections_global)
    
    ''' from custom_evaluation.py
        added average precision 16/03'''
    
    def on_epoch_end(self, epoch, logs={}):

        if epoch % self._period == 0 and self._period != 0:
            precision,recall,f1score,_map, average_precisions = self.evaluate_map()
            print('\n')
            for label, average_precision in average_precisions.items():
                print(self._yolo.labels[label], '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(_map))

            if self._save_best and self._save_name is not None and _map > self.bestMap:
                print("mAP improved from {} to {}, saving model to {}.".format(self.bestMap, _map, self._save_name))
                self.bestMap = _map
                self.model.save(self._save_name)
            else:
                print("mAP did not improve from {}.".format(self.bestMap))

            if self._tensorboard is not None:
                with summary_ops_v2.always_record_summaries():
                    with self._tensorboard._val_writer.as_default():
                        name = "mAP"  # Remove 'val_' prefix.
                        summary_ops_v2.scalar('epoch_' + name, _map, step=epoch)

    def evaluate_map(self):
        precisions,recalls,f1_scores,average_precisions = self._calc_avg_precisions()
        _map = sum(average_precisions.values()) / len(average_precisions)

        return precisions,recalls,f1_scores,_map, average_precisions

    def _calc_avg_precisions(self):
        # gather all detections and annotations
        # all_detections = [[None for _ in range(self._generator.num_classes())]
        #                   for _ in range(self._generator.size())]
        # all_annotations = [[None for _ in range(self._generator.num_classes())]
        #                    for _ in range(self._generator.size())]
        all_detections = [[[] for _ in range(self._generator.num_classes())]
                           for _ in range(self._generator.size())]
        all_annotations = [[[] for _ in range(self._generator.num_classes())]
                           for _ in range(self._generator.size())]
        

        for i in range(self._generator.size()):
            raw_image, img_name = self._generator.load_image(i)
            raw_height, raw_width, _ = raw_image.shape  

            # make the boxes and the labels
            # if i % 50 == 0 : 
            #     print(f"prediction number {i} done")
            print(f"prediction number {i} done")
            pred_boxes = self._yolo.predict(raw_image,
                                            iou_threshold=self._iou_threshold,
                                            score_threshold=self._score_threshold)

            score = np.array([box.score for box in pred_boxes])
            
            if len(score) != 0:
                print('score ', score)
            pred_labels = np.array([box.get_label() for box in pred_boxes])
            if len(pred_labels) != 0:
                print('pred label ', pred_labels)
            if len(pred_boxes) > 0:
                print(pred_boxes)
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
                print('pred boxes ',pred_boxes)
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(self._generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = self._generator.load_annotation(i)
            
            if annotations.shape[1] > 0:
                # copy ground truth to all_annotations
                for label in range(self._generator.num_classes()):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
            print("all ann ", all_annotations[i])
        # print('all_detections ', all_detections)
        # print('all_annotations ', all_annotations)
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        precisions = {}
        recalls = {}
        f1_scores = {}

        for label in range(self._generator.num_classes()):
            print("Calculation on label: ", label)
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(self._generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += len(annotations)
                detected_annotations = []
                if len(detections) != 0: 
                    print(f"detections {detections} \n label {label}")
                    print(f"annotations {annotations}")


                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= self._iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
            f1_score = 2*precision * recall/(precision + recall)
            print(f"label {label}, precision {precision}, recall {recall}, f1_score {f1_score}")
            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision
            precisions[label] = precision
            recalls[label] = recall
            f1_scores[label] = f1_score

        print('Computing done')
        
        return precisions,recalls,f1_scores,average_precisions
