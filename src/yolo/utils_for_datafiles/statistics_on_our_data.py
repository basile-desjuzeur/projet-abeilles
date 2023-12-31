#Explication des dernières mises à jour
#Ce code nous permet de constituer un ensemble de validation composé seulement d'images provenant des 
# tasks. Si l'espèce possède plus de 20 images alors on cappe à 20 et si elle en possède moins alors 
# on cappe 
#à 4 pour laissé des images des tasks dans le train.

import csv
import json 
import argparse

argparser=argparse.ArgumentParser("Statistics on data")

argparser.add_argument(
    '-c',
    '--conf',
    default='/src/config/bees_detection.json',
    help='path to configuration file'
)

def _main_(args):

    config_path=args.conf

    with open(config_path) as config_buffer:
        config=json.loads(config_buffer.read())

    path_to_train = config['data']['train_csv_file']
    path_to_validation = config['data']['valid_csv_file']
    path_to_test = config['data']['test_csv_file']
    labels = config['model']['labels']

    #on veut le nombre d'image par espèce

    counter_image_per_espece_train = {}
    for label in labels:
        counter_image_per_espece_train[label] = 0

    with open(path_to_train, 'r') as file_buffer:
        reader = csv.reader(file_buffer, delimiter=',')
        next(reader)
        train = list(reader) #renvoie une liste de listes où chaque liste est une ligne du csv délimitée par des virgules
        for line in train:
            counter_image_per_espece_train[line[5]] += 1

    counter_image_per_espece_valid = {}
    for label in labels:
        counter_image_per_espece_valid[label] = 0

    with open(path_to_validation, 'r') as file_buffer:
        reader = csv.reader(file_buffer, delimiter=',')
        next(reader)
        train = list(reader) #renvoie une liste de listes où chaque liste est une ligne du csv délimitée par des virgules
        for line in train:
            counter_image_per_espece_valid[line[5]] += 1

    print(f"Nombre d'images par espèce dans le train:", counter_image_per_espece_train) 
    print("\n")
    print(f"Nombre d'images par espèce dans la validation:", counter_image_per_espece_valid)  
    print("\n")
    print("--------------------------------------------------------------------------------------")
    #on veut le nombre d'images qui viennet des tasks par espèce

    counter_task_train = {}
    for label in labels:
        counter_task_train[label] = 0

    with open(path_to_train, 'r') as file_buffer:
        reader = csv.reader(file_buffer, delimiter=',')
        next(reader)
        train = list(reader) #renvoie une liste de listes où chaque liste est une ligne du csv délimitée par des virgules
        for line in train:
            if 'task' in line[0]:
                espece = line[5]
                counter_task_train[espece] += 1

    counter_task_valid = {}
    counter_iNat_valid = {}

    for label in labels:
        counter_task_valid[label] = 0
        counter_iNat_valid[label] = 0

    with open(path_to_validation, 'r') as file_buffer:
        reader = csv.reader(file_buffer, delimiter=',')
        next(reader)
        train = list(reader) #renvoie une liste de listes où chaque liste est une ligne du csv délimitée par des virgules
        for line in train:
            if 'task' in line[0]:
                espece = line[5]
                counter_task_valid[espece] += 1
            else:
                espece = line[5]
                counter_iNat_valid[espece] += 1

    faible = []
    for especes in counter_task_train:
        if counter_task_train[especes] < 300:
            faible.append(especes)


    print("\n")
    print("Nombre d'images provenant des tasks dans le train:", counter_task_train)
    print("\n")
    print("Classes avec un faible rapport task/iNat dans le train:", faible)
    print("\n")
    print("Nombre d'images provenant des tasks dans la validation:", counter_task_valid)
    print("\n")
    print("Nombre d'images provenant des iNat dans la validation:", counter_iNat_valid)
    #on veut le nombre d'images par espèce par task

if __name__=='__main__':
    _args=argparser.parse_args()
    _main_(args=_args)