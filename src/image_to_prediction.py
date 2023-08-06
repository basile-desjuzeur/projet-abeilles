import argparse
import pandas as pd 
import numpy as np
import os

from yolo import predict as predict_yolo
from classification import predict as predict_classification


argparser = argparse.ArgumentParser(description='Predict bee images pipeline')


argparser.add_argument(
    '-i',
    '--input',
    required=True, 
    help = """Path to input with images to download,
            """  
                       )



argparser.add_argument(
    '-yc',
    '--yolo_config',
    help='Path to yolo config file',
    required=True,
    default='../../datafiles/yolo/configs/final_config.json')


argparser.add_argument(
    '-yw',
    '--weights',



    help='Path to yolo weights file',)


class Predictor:
    
    def __init__(self, yolo_config, yolo_weights):
        self.yolo_config = yolo_config
        self.yolo_weights = yolo_weights

