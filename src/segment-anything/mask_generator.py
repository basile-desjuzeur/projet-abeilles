import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json 

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry



parser = ArgumentParser("Generate bbox from an image using SAM")

parser.add_argument("--sam_checkpoint", type=str, default="/home/basile/Documents/projet_bees_detection_basile/bees_detection/src/segment-anything/checkpoints/sam_vit_b_01ec64.pth")

parser.add_argument("--model_type", type=str, default="vit_b")

parser.add_argument("--device", type=str, default="cuda")

parser.add_argument(
    "--img_path",
    type=str,
    default='/home/basile/Documents/projet_bees_detection_basile/test2/Amegilla_garrula_female.jpeg'
)

parser.add_argument(
    '--output',
    type=str,
    default='/home/basile/Documents/projet_bees_detection_basile/test3')

parser.add_argument(
    '-c',
    '--conf',
    default='/home/basile/Documents/projet_bees_detection_basile/bees_detection/src/yolo/config/bees_detection.json',
    help='path to configuration file')



#### utils functions ####


def show_mask(mask, ax, random_color=False):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def correspond_to_anchors(anchors, bbox, threshold=0.5):
    """
    Check if the bbox correspond to one of the anchors

    params:
        anchors: list of anchors defined in the config file (list of pairs (width, height))
        bbox: bounding box
        threshold: threshold to consider that the bbox correspond to an anchor

    return:
        True if the bbox correspond to one of the anchors, False otherwise    
    """ 
    
    for anchor in anchors:
        if bbox[2] > anchor[0] - threshold and bbox[2] < anchor[0] + threshold and bbox[3] > anchor[1] - threshold and bbox[3] < anchor[1] + threshold:
            return True
    return False


##### predict #####



def main(args): 

    # Initialize the model
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(sam, output_mode='coco_rle')

    # Load the image
    img_path = args.img_path
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to 500x333
    initial_shape = image.shape
    image = cv2.resize(image, (500,333))


    # Generate the masks
    masks = mask_generator.generate(image)


    # Loads the anchors
    with open(args.conf) as f:
        config = json.load(f)

    anchors = config['model']['anchors']

    # Create the csv files
    other_csv = os.path.join(args.output, 'other.csv')
    output_csv = os.path.join(args.output, 'output.csv')


    for i,mask in enumerate(masks):

        bbox = mask['bbox']
        bbox = [str(int(x)) for x in bbox]

        # Reshape the bounding box to the initial shape
        bbox[0] = str(int(int(bbox[0]) * initial_shape[1] / 500))
        bbox[1] = str(int(int(bbox[1]) * initial_shape[0] / 333))
        bbox[2] = str(int(int(bbox[2]) * initial_shape[1] / 500))
        bbox[3] = str(int(int(bbox[3]) * initial_shape[0] / 333))


        label = 'bee'
        score = str(mask['stability_score'])
        predicted_iou = str(mask['predicted_iou'])

        # Check if the bounding box correspond to one of the anchors
        is_anchor = correspond_to_anchors(anchors, bbox)

        with open(output_csv, 'a') as f:
            f.write(img_path + ',' + ','.join(bbox) + ',' + label + ',' + str(initial_shape[0])+','+ str(initial_shape[1])+'\n')

        with open(other_csv, 'a') as f:
            f.write(img_path + ',' + ','.join(bbox) + ',' + label + ',' + score+','+predicted_iou+','+str(is_anchor)+'\n')


            


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

