import pandas as pd 
import numpy as np
from argparse import ArgumentParser
import json
import os 
import sklearn

argparser=ArgumentParser("Splits the desired csv file to 3 different files for training/validation/test")

argparser.add_argument(
    '-c',
    '--conf',
    default='/src/config/bees_detection.json',
    help='path to configuration file')

def _main_(args):

    config_path=args.conf

    with open(config_path) as config_buffer:
        config=json.loads(config_buffer.read())

    path_to_dataset=config['data']['complete_csv_file']
    
    # proportion of the splits of the whole dataset

    proportion_train=0.8
    proportion_val=0.1
    
    
    df=pd.read_csv(path_to_dataset)

    train, val, test = np.split(df.sample(frac=1, random_state=42), 
                       [int(proportion_train*len(df)), int((proportion_train+proportion_val)*len(df))])

    # df to csv

    path=(os.sep).join(path_to_dataset.split(os.sep)[:-1])
    df.to_csv()
    train.to_csv(os.path.join(path,'train.csv'),header=False,index=False)
    val.to_csv(os.path.join(path,'validation.csv'),header=False,index=False)
    test.to_csv(os.path.join(path,'test.csv'),header=False,index=False)
    

if __name__=='__main__':
    _args = argparser.parse_args()
    _main_(args=_args)
