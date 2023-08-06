import os 
from pathlib import Path
import pandas as pd 
import json 
from PIL import Image
import argparse

argparser = argparse.ArgumentParser(
    description='Convert json files to csv')

argparser.add_argument(
    '-j',
    '--jsons_path',
    help='path to json files')

argparser.add_argument(
    '-i',
    '--images_path',
    help='path to images folder')

argparser.add_argument(
    '-t',
    '--taxon',
    help='taxon to detect')

argparser.add_argument(
    '-o',
    '--output_path',
    help='path to output csv file')


def json_to_csv(path_to_json,path_to_images,taxon_detection,path_to_output):

    master_dict=[]      # dict to be converted in csv
    minor_dict=[]       # dict with the corrupted files


    # Create a dataframe with the info of every pictures

    for file in os.listdir(path=path_to_json):

        with open(file=os.path.join(path_to_json,file)) as file:

            json_file=json.load(file)
            json_file_verified=[picture for picture in json_file if picture['visited']==1] #only takes verified pic
            
            for picture in json_file_verified:
                for box in picture['boxes']:

                    file_path=picture['file_path'] 
                    xmin=box['xmin']
                    xmax=box['xmax']
                    ymin=box['ymin']
                    ymax=box['ymax']
                    taxon=picture['specie']


                    # getting the image size 
                    # in the json the path is "./BD_71/Amegilla quadrifasciata/Amegilla quadrifasciata51495.jpg"
                    path_to_img=str(file_path)[7:]
                    path_to_img=Path(path_to_images + path_to_img)

                    try:
                        img=Image.open(path_to_img) 
                        w,h= img.size
                        box_info=[file_path,xmin,ymin,xmax,ymax,taxon,w,h,]
                        master_dict.append(box_info)
                    
                    except Exception:

                        minor_dict.append(path_to_img)

    df_ok=pd.DataFrame(master_dict,columns=['file_path','xmin','ymin','xmax','ymax','taxon','w','h'])


    # Cleans the dataframe so that it can be processed by gen_anchors.py

    df_ok['xmin']=df_ok['xmin']*df_ok['w']       # xmin,.. stored as relative values in json
    df_ok['xmax']=df_ok['xmax']*df_ok['w']       # need to be converted to absolute values
    df_ok['ymin']=df_ok['ymin']*df_ok['h']       # for gen_anchors    
    df_ok['ymax']=df_ok['ymax']*df_ok['h']

    df_ok['taxon']=taxon_detection                   # we only need one discrimination
    df_ok.iloc[:,0]=df_ok.iloc[:,0].map(lambda row : row.split('/',1)[1])

    df_missing=pd.DataFrame(minor_dict)

    # Dataframe to csv
    # df_missing allows you to check if there is no corrupted pictures     
    path_1 = os.path.join(path_to_output,'BD_71_input.csv')
    path_2 = os.path.join(path_to_output,'BD_71_missing.csv')            

    df_ok.to_csv(path_1,index=False)
    df_missing.to_csv(path_2,index=False)


if __name__ == "__main__":
        
        args = argparser.parse_args()
    
        json_to_csv(
            path_to_json=args.jsons_path,
            path_to_images=args.images_path,
            taxon_detection=args.taxon,
            path_to_output=args.output_path
        )