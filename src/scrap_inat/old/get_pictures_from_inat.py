import pandas as pd
import sqlite3
import os
import shutil
import asyncio

from utils_download_from_inat import download_from_csv_not_asynchrone,download_from_csv

########## INPUTS ##########

# You must provide an iterable with all the taxon names you want to download the images from
# stored in the df_taxa_in_bdd variable

# path to the csv file with all the taxon name
csv_path = '/home/basile/Documents/projet_bees_detection_basile/bees_detection/src/datafiles/classification/hierarchy.csv'   
# Load the csv file with all the taxon name
df_taxa_in_bdd = pd.read_csv(csv_path, sep=',')
#  Get the taxon names as an iterable
df_taxa_in_bdd = df_taxa_in_bdd['species']



# path to the sqlite database
sqlite_path = '/home/basile/Documents/projet_bees_detection_basile/data_bees_detection/inat_12_04/inat.db'

# path to the folder where the csv and the images will be saved
output_folder = '/home/basile/Documents/projet_bees_detection_basile/data_bees_detection/inat_25_04'

# path to the csv qith all the already downloaded images
path_whole_dataset = '/home/basile/Documents/projet_bees_detection_basile/bees_detection/src/datafiles/crop/25_04/files_in_whole_dataset_with_real_labels/whole_dataset.csv'

# get the names of the images already downloaded as an iterable
df_whole_dataset = pd.read_csv(path_whole_dataset, sep=',')
df_whole_dataset = df_whole_dataset['Paths']
df_whole_dataset = df_whole_dataset.str.split('/').str[-1]


########## UTILS ##########


def name_to_taxon_id(taxon_name, c):

    # Execute the query
    c.execute("select taxon_id from taxa where name= ?", (taxon_name,))

    # Fetch the results
    result = c.fetchone()

    return result[0]


def info_from_taxon_id(taxon_id, c):
    """
    Return the taxon_id, photo_id, extension and observation_uuid of the taxon_id
    as a dataframe
    """

    # Execute the query
    c.execute("SELECT taxon_id, photo_id, extension, photos.observation_uuid FROM observations INNER JOIN photos on photos.observation_uuid = observations.observation_uuid where taxon_id = ?", (taxon_id,))

    # Fetch the results
    result = c.fetchall()

    # Create a dataframe with the result
    df = pd.DataFrame(
        result, columns=['taxon_id', 'photo_id', 'extension', 'observation_uuid'])
    
    # Drop the duplicates
    df = df.drop_duplicates()


    return df


def is_already_downloaded(df_taxon, df_dataset):   
    """
    Checks if the images of the taxon are already downloaded
    :param df_taxon: dataframe with the information of the taxon fetched from the sqlite database
                    # taxon_id, photo_id, extension, observation_uuid

    :param df_dataset: dataframe with all the names of the images already downloaded
                    # name (e.g. Apis mellifera.jpg)

    :return: a dataframe with the images that are not already downloaded
    """

    # Get the names of the images
    df_taxon['name'] = df_taxon['photo_id'].astype(str) + '.' + df_taxon['extension']

    # Check if the images are already downloaded
    df_taxon['is_already_downloaded'] = df_taxon['name'].isin(df_dataset)

    # Get only the images that are not already downloaded
    df_taxon = df_taxon[df_taxon['is_already_downloaded'] == False]

    # Drop the column is_already_downloaded
    df_taxon = df_taxon.drop('is_already_downloaded', axis=1)

    # Drop the column name 
    df_taxon = df_taxon.drop('name', axis=1)

    return df_taxon


########## MAIN ##########

def main():

    # Connect to the database
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # Output folders
    path_to_csv_files = os.path.join(output_folder, 'csv_files')
    path_to_img_files = os.path.join(output_folder, 'Pictures')
    
    # Deletes csv and images files if they already exist
    if os.path.exists(path_to_csv_files):
        shutil.rmtree(path_to_csv_files)
    if os.path.exists(path_to_img_files):
        shutil.rmtree(path_to_img_files)

    # Creates output folders
    os.mkdir(path_to_csv_files)
    os.mkdir(path_to_img_files)



    errors = []

    # For each taxon name
    for taxon_name in df_taxa_in_bdd:

        print('Downloading the images of the taxon: ' + taxon_name + '...')

        # Get the taxon id

        try:
            taxon_id = name_to_taxon_id(taxon_name, c)

        except TypeError:

            smiley_error = u'\u274C'
            print('No taxon id for the taxon: ' +
                  taxon_name + '  ' + smiley_error + '\n')
            errors.append(taxon_name)
            continue

        # Get the info from the taxon id
        df_info_taxon = info_from_taxon_id(taxon_id, c)

        # Check if the images are already downloaded
        df_info_taxon = is_already_downloaded(df_info_taxon, df_whole_dataset)

        # Check if there are new images
        if df_info_taxon.empty:
            smiley_error = u'\u274C'
            print('No new images for the taxon: ' +
                  taxon_name + '  ' + smiley_error + '\n')
            continue

        # Save the infos in a csv file
        df_info_taxon.to_csv(os.path.join(path_to_csv_files,taxon_name+'.csv'), index=False)

        # Download the images and wait until the download is finished
        # download_from_csv_not_asynchrone(os.path.join(path_to_csv_files,taxon_name+'.csv'), taxon_name, images_folder=path_to_img_files)
        download_from_csv(os.path.join(path_to_csv_files,taxon_name+'.csv'), taxon_name, images_folder=path_to_img_files)
    

        smiley_done = u'\u2705'
        print('Done for the taxon: ' + taxon_name + '  ' + smiley_done + '\n')

    print('The following taxon names have not been found in the database: ')
    for error in errors:
        print(error + '\n')

    # Close the connection
    conn.close()

    # # Remove the csv files
    # shutil.rmtree(path_to_csv_files)


if __name__ == "__main__":
    main()

    a