from evaluate import _main_ as evaluate
import argparse
import os 
import tensorflow as tf
import pandas as pd 
import shutil 
import json 
from datetime import datetime
from matplotlib import pyplot as plt

'''
This scripts aims to evaluate the model for each species separately
Since the model is only trained on one "Mega-specie" (e.g. Anthophila)
we evaluate create one csv file per species and evaluate the model on each of them
returning a csv file with the precision, recall, F1 score and mAP for each species 
+ for all the species together

# pas super opti mais ca marche
'''


argparser=argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf', 
    help='Path to configuration file',
    default='/src/datafiles/yolo/configs/benchmark_configbees_detection_copy.json'
)

argparser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to pretrained weights')

argparser.add_argument( 
  '-l',
  '--lite',
  default='',
  type=str,
  help='Path to tflite model')

argparser.add_argument(
    '-f',
    '--file',
    default='',
    help='path to file to evaluate')


def _main_eval_diff_(args): 


    start_time=datetime.now()

    config_path=args.conf

    with open(config_path,'r+') as config_buffer:   
        config = json.loads(config_buffer.read())


    #1) get the images to evaluate 
    
    if args.file == '':
        files_to_evaluate_path=config['data']['test_csv_file']
    else:
        files_to_evaluate_path=args.file

    #test if the files exists

    if type(files_to_evaluate_path)==str:
        files_to_evaluate_path=[files_to_evaluate_path]
        final_csv_name=files_to_evaluate_path[0].split(os.path.sep)[-1].split('.')[0]
    else:
        files_to_evaluate_path=[file_to_evaluate_path for file_to_evaluate_path in files_to_evaluate_path if os.path.exists(file_to_evaluate_path)]
        final_csv_name='__'.join([file_to_evaluate_path.split(os.path.sep)[-1].split('.')[0] for file_to_evaluate_path in files_to_evaluate_path])


    if files_to_evaluate_path==[]:
        print('No file to evaluate')
        return


    df_evaluate=pd.concat([pd.read_csv(file_to_evaluate_path,header=None) for file_to_evaluate_path in files_to_evaluate_path])

    # get the labels
    # we assume that filenames follow this format : /path/to/dataset/label/filename.jpg
    labels=df_evaluate[0].str.split(os.path.sep,expand=True).iloc[:,-2].unique() 


    #2) create a temp folder

        
    os.makedirs('/src/datafiles/yolo/temp_'+start_time.strftime("%Y%m%d-%H%M%S"))

    #3) stores in this folder one csv per label + all the species together

    evaluate_paths=[]       # to store the path of the csv to be written on config['data']['test_csv_file']

    for label in labels: 

        temp_df=df_evaluate.loc[df_evaluate[0].str.split(os.path.sep,expand=True).iloc[:,-2]==label]
        label=label.replace(' ','_')
        path_to_csv='/src/datafiles/yolo/temp_'+start_time.strftime("%Y%m%d-%H%M%S")+'/temp_csv_{}.csv'.format(label)
        temp_df.to_csv(path_to_csv,index=False,header=False)
        evaluate_paths.append(path_to_csv)

    # create a csv with all the pictures coming from the dataset
    path_to_csv='/src/datafiles/yolo/temp_'+start_time.strftime("%Y%m%d-%H%M%S")+'/temp_csv_all_species.csv'
    df_evaluate.to_csv(path_to_csv,index=False,header=False)
    evaluate_paths.append(path_to_csv)

    previous_test_csv_file=config['data']['test_csv_file']              # we want to keep the previous test csv file for 8)


    #4) on each csv file, change the config file to only evaluate on this species and evaluate the model


    for evaluate_path in evaluate_paths:

        #4.1) change the config file to only evaluate on this species

        with open(config_path) as config_buffer:   
            config = json.loads(config_buffer.read())


        config['data']['test_csv_file']=evaluate_path

        with open(config_path,'w') as config_buffer:
            config=json.dump(config,config_buffer,indent=4)

        #4.2) evaluate the model

        _args=argparser.parse_args()
        _args.conf=config_path

        if args.weights != '':
            _args.weights=args.weights

        if args.lite != '':
            _args.lite=args.lite

        with open(_args.conf) as config_buffer:
            config = json.loads(config_buffer.read())


        class_p_global,class_r_global,class_f1_global,bbox_p_global,bbox_r_global,bbox_f1_global, ious_global,intersections_global=evaluate(_args)


        #4.3) store the metrics in a dataframe


        label=''.join(evaluate_path.split(os.path.sep)[-1].split('_')[-2:]).split('.')[0]
        label=label.replace('_','  ')

        if evaluate_path==evaluate_paths[0]:
            df=pd.DataFrame({'label':[label],'class_p_global':[class_p_global],'class_r_global':[class_r_global],'class_f1_global':[class_f1_global],'bbox_p_global':[bbox_p_global],'bbox_r_global':[bbox_r_global],'bbox_f1_global':[bbox_f1_global],'ious_global':[ious_global],'intersections_global':[intersections_global]})
        else:
            new_df=pd.DataFrame({'label':[label],'class_p_global':[class_p_global],'class_r_global':[class_r_global],'class_f1_global':[class_f1_global],'bbox_p_global':[bbox_p_global],'bbox_r_global':[bbox_r_global],'bbox_f1_global':[bbox_f1_global],'ious_global':[ious_global],'intersections_global':[intersections_global]})
            df=pd.concat([df,new_df],axis=0,ignore_index=True)

       

    end_time=datetime.now()

    ############################################################

    #                       CLEANING AND SAVING

    ############################################################

    #5) restore the previous test csv file

    with open(config_path) as config_buffer:   
        config = json.loads(config_buffer.read())   

    config['data']['test_csv_file']=previous_test_csv_file


    with open(config_path,'w') as config_buffer:
        config=json.dump(config,config_buffer,indent=4)

    
    #6) delete the temp folder

    shutil.rmtree('/src/datafiles/yolo/temp_'+start_time.strftime("%Y%m%d-%H%M%S"))

    #7) save the dataframe to a csv 

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())   

    path_evaluates=config['data']['evaluations_path']
    current_time=datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    df.to_csv(os.path.join(path_evaluates,'evaluation_on_{}_on_{}.csv'.format(final_csv_name,current_time)),header=True,index=False)

  
    #9) delete the pickles files created during the evaluation

    pickles_path=config['data']['saved_pickles_path']
    for folder in os.scandir(pickles_path):

        folder_time=folder.name.split('_')[1]
        folder_time=datetime.strptime(folder_time,'%Y-%m-%d-%H:%M:%S')

        # if the folder has been created during the evaluation, we delete it
        if start_time<=folder_time:
            # except if it contains the csv file with all the species
            if 'temp_csv_all_species' not in os.listdir(folder.path):
                shutil.rmtree(folder.path)

    return df

if __name__ == '__main__':
    _args = argparser.parse_args()
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    with tf.device('/GPU:' + gpu_id):
        _main_eval_diff_(_args)
