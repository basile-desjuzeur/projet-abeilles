import pandas as pd
import cv2 as cv 
import numpy as np
import tqdm 

PATH_TO_DATASET ="/src/datafiles/final_datafiles/dataset_yolo_cropped_with_cleaned_structure.csv"
    

def get_means_and_stds_for_normalization(path_to_dataset):
    """
    Outputs the means and stds of the images in the dataset for normalization
    
    Parameters
    ----------
    path_to_dataset : str
        Path to the dataset file (.csv)
        # path , # label

    Returns
    -------
    means : list
        List of the means of the images in the dataset
        RGB order
    stds : list
        List of the stds of the images in the dataset
        RGB order
    """

    df = pd.read_csv(path_to_dataset)

    means = []
    stds = []

     # Compute the mean and std of each image in the dataset
    for path in tqdm.tqdm(df['Paths']):
       
        img = cv.imread(path)
        means.append(img.mean(axis=(0,1)))
        stds.append(img.std(axis=(0,1)))


    # Compute the mean and std of the dataset

    means = np.array(means)
    stds = np.array(stds)

    means = list(means.mean(axis=0))
    stds = list(stds.mean(axis=0))

    # Print the means and stds
    print('-'*50)

    print("Means : ")
    print('Red : ', means[0], 'Green : ', means[1], 'Blue : ', means[2])

    print('\n')

    print("Stds : ")
    print('Red : ', stds[0], 'Green : ', stds[1], 'Blue : ', stds[2])

    print('-'*50)


    return means, stds
    

if __name__ == "__main__":

    get_means_and_stds_for_normalization(PATH_TO_DATASET)