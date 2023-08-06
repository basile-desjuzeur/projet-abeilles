import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from segment_anything import build_sam, SamPredictor, sam_model_registry
from segment_anything.utils.amg import MaskData, mask_to_rle_pytorch, batched_mask_to_box

#####  parameters #####


csv_path = "/home/basile/Documents/projet_bees_detection_basile/test/detected_images_2023-04-19_16-55.csv"

sam_checkpoint = "/home/basile/Documents/projet_bees_detection_basile/bees_detection/src/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

output_path = "/home/basile/Documents/projet_bees_detection_basile/test"

#####  util functions #####

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=10))

#####  predict #####

def predict(img_path, sam_checkpoint, model_type, device, output_path,input_point ): #box ,,

    image_name = (img_path.split(os.path.sep)[-1]).split('.')[0]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = input_point
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_labels=input_label,
        multimask_output=False,

    )
    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(100,100))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     show_box(box, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('on')
    #     plt.savefig(f"{output_path}/"+ image_name + ".png")

    return masks    


def read_csv(csv_path):

    with open(csv_path, "r") as f:

        img_paths = []
        img_bbox = []
        img_bbox_centers = []

        for line in f:
            line = line.split(",")

            img_path = line[0]
            img_paths.append(img_path)

            bbox= np.array([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
            img_bbox.append(bbox)

            bbox_center = np.array([[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]])
            img_bbox_centers.append(bbox_center)

    return img_paths, img_bbox, img_bbox_centers


if __name__ == "__main__":


    img_paths, img_bbox, img_bbox_centers = read_csv(csv_path)

    for i, img_path in enumerate(img_paths):

        print( "Processing image n°", i, ":", img_path.split(os.path.sep)[-1])
        bbox_center = img_bbox_centers[i]
        bbox = img_bbox[i]
        mask_i = predict(img_path, sam_checkpoint, model_type, device, output_path,bbox_center)#, box= bbox
        
    
        mask_i = torch.from_numpy(mask_i)


        mask_i = batched_mask_to_box(mask_i)
        print(mask_i)

  
        xmin, ymin, xmax, ymax = int(mask_i[0][0]), int(mask_i[0][1]), int(mask_i[0][2]), int(mask_i[0][3])

        print("Image n°", i, ":", img_path.split(os.path.sep)[-1], "processed")

        label = "bee"

        img_w, img_h = cv2.imread(img_path).shape[1], cv2.imread(img_path).shape[0]



        with open(f"{output_path}/"+ "test_for_SA.csv", "a") as f:
            f.write(img_path + "," + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + label + "," + str(img_w) + "," + str(img_h) +"\n")

        