import os
import csv
import wget
from tqdm import tqdm
import asyncio
import sqlite3


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


@background
def get_image(image_url, target_dest):
    wget.download(image_url, target_dest)
    return

def get_image_not_asynchrone(image_url, target_dest):
    wget.download(image_url, target_dest)
    return

def download_from_csv(path_to_csv, taxon_name, images_folder):
    """
    Download images from a csv file
    :param path_to_csv: path to the csv file
            csv file must have the following format:
            #taxon_id	#photo_id #extension #observation_uuid
    :param taxon_name: name of the taxon
    :param images_folder: path to the folder where the images will be saved
    :return:
    """

    # Create the folder if it does not exist
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    #  Load CSV of selected pictures : #taxon_id	#photo_id #extension #observation_uuid
    with open(path_to_csv, newline='') as csvfile:
        lines = csvfile.read().split("\n")
        for i, row in enumerate(tqdm(lines)):
            data = row.split(',')
            if i > 0 and len(data) > 2:
                taxon_id = data[0]
                photo_id = data[1]
                extension = data[2]

                if not os.path.exists(os.path.join(images_folder, taxon_name)):
                    os.mkdir(os.path.join(images_folder, taxon_name))

                image_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.{extension}"
                image_name = photo_id + '.' + extension
                target_dest = os.path.join(
                    images_folder, taxon_name, image_name)
                get_image(image_url, target_dest)

def download_from_csv_not_asynchrone(path_to_csv, taxon_name, images_folder):
    """
    // Same function as download_from_csv but without asyncronous //
    Download images from a csv file
    :param path_to_csv: path to the csv file
            csv file must have the following format:
            #taxon_id	#photo_id #extension #observation_uuid
    :param taxon_name: name of the taxon
    :param images_folder: path to the folder where the images will be saved
    :return:
    """

    # Create the folder if it does not exist
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    #  Load CSV of selected pictures : #taxon_id	#photo_id #extension #observation_uuid
    with open(path_to_csv, newline='') as csvfile:
        lines = csvfile.read().split("\n")
        for i, row in enumerate(tqdm(lines)):
            data = row.split(',')
            if i > 0 and len(data) > 2:
                taxon_id = data[0]
                photo_id = data[1]
                extension = data[2]

                if not os.path.exists(os.path.join(images_folder, taxon_name)):
                    os.mkdir(os.path.join(images_folder, taxon_name))

                image_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.{extension}"
                image_name = photo_id + '.' + extension
                target_dest = os.path.join(
                    images_folder, taxon_name, image_name)
                get_image_not_asynchrone(image_url, target_dest)

