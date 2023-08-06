import os 
import pandas as pd
import tqdm 
from PIL import Image
import argparse
import cv2




#### Example of use ####

# python3 cropfromcsv.py -s /home/basile/Documents/projet_bees_detection_basile/data_bees_detection/whole_dataset 
# -t /home/basile/Documents/projet_bees_detection_basile/data_bees_detection/whole_dataset_cropped 
# -c /home/basile/Documents/projet_bees_detection_basile/bees_detection/src/datafiles/crop/predict_csv/other/whole_datast_predicted_latest.csv

########################

argparser = argparse.ArgumentParser(description='Crop images from csv file')

argparser.add_argument('-s',
                          '--source_path',
                          help='Source path where raw images are stored')

argparser.add_argument('-t',
                            '--target_path',
                            help='Target path where cropped images will be stored')



argparser.add_argument('-k',
                       '--keep_folder_structure',
                          help='Keep the folder structure of the source path i.e source/path/label/image.jpg',
                          default=True)

argparser.add_argument('-c',
                            '--csv_path',
                            help='Path to the csv file outputed by predict.py')


#### Script ####


def crop_image(img_path, x, y, w, h):
    '''
    Crop image from x, y, w, h coordinates
    '''
    img = Image.open(img_path)
    img = img.crop((x, y, x+w, y+h))
    
    return img 


def crop_images(source_path, target_path, csv_path, keep_folder_structure):
    """
    Crop images from csv file output of predict.py and save them in target_path

    Parameters
    ----------
    source_path : str
        Path to the folder containing the images to crop
        We assume that the hierarchy is as follows:
        path/to/source_path/label/image.jpg

    target_path : str
        Path to the folder where the cropped images will be saved
        Hierachy will be the same as source_path

    csv_path : str
        Path to the csv file outputed by predict.py
        Format of the csv file:
        # file_path, xmin, ymin, xmax, ymax, class_name, width, height
    
    keep_folder_structure : bool
        If True, the folder structure of source_path will be kept in target_path
        If False, all the images will be saved in target_path

        
    Returns
    -------
    None
    """

    # Read the csv file
    df = pd.read_csv(csv_path)

    ##  Create all the folders ##

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Get the folders of labels
    df_folders = df.iloc[:,0].apply(lambda x: x.split('/')[:-1])
    df_folders = df_folders.apply(lambda x: '/'.join(x))
    df_folders = df_folders.drop_duplicates()
    df_folders = df_folders.apply(lambda x: x.replace(source_path, target_path))

    if not os.path.exists(target_path):
        os.makedirs(target_path)


    for folder in df_folders:
        if not os.path.exists(folder):

            try : 
                os.makedirs(folder)
            except :
                print('problem with : {}'.format(folder))
                continue


    # Crop the images
    for index, row in tqdm.tqdm(df.iterrows()):

        # Get the image path
        img_path = row[0]

        if keep_folder_structure:

            # Get the folder of the image
            label_name = img_path.split('/')[:-2]
            img_name = img_path.split('/')[-1]

            new_folder = os.path.join(target_path, '/'.join(label_name))
            new_img_path = img_path.replace(source_path, target_path)


            new_folder = new_img_path.split('/')[:-1]
            new_folder = '/'.join(new_folder)

            # if not os.path.exists(new_folder):
            #     os.makedirs(new_folder)

            new_img_path = os.path.join(new_folder, img_name)

        else :
            break

     
        # Get the coordinates
        x = row[1]
        y = row[2]
        w = row[3]
        h = row[4]

        # Crop the image
        img = crop_image(img_path, x, y, w, h)

        # Save the image
        img.save(new_img_path)


if __name__ == '__main__':
    args = argparser.parse_args()
    source_path = args.source_path
    target_path = args.target_path
    csv_path = args.csv_path
    keep_folder_structure = args.keep_folder_structure

    crop_images(source_path, target_path, csv_path, keep_folder_structure)
    