import os
import json
from PIL import Image
import cv2
import argparse


argparser = argparse.ArgumentParser(
    description='Crop images from json files')

argparser.add_argument(
    '-j',
    '--jsons_path',
    help='path to json files')

argparser.add_argument(
    '-o',
    '--output_path',
    help='path to output folder')

argparser.add_argument(
    '-b',
    '--bd71',
    help='path to bd_71 folder')


def crop_image(img,box,w,h):

    xmin = int(box['xmin']*w)
    xmax = int(box['xmax']*w)
    ymin = int(box['ymin']*h)   
    ymax = int(box['ymax']*h)

    #crop image
    if xmin<0: 
        xmin = 0
    if ymin<0:
        ymin = 0
    if xmax>w:
        xmax = w
    if ymax>h:
        ymax = h
    
    if xmin == xmax or ymin == ymax:
        img2 = img

    else:
        # crop image with PIL
        img = Image.fromarray(img)
        img2 = img.crop((xmin,ymin,xmax,ymax))


    return img2


def crop_from_json(json_path,output_path,bd71_path):

    '''
    We consider that Json is structured as following : 
    and that there is only one specie per json file
    {"file_path":"./BD_71/Amegilla quadrifasciata/Amegilla quadrifasciata51495.jpg",
    "area":152320.3125,
    "boxes":[
        {"xmin":0.6175,"ymin":0.2584745762711864,"xmax":0.975,"ymax":0.7923728813559322},
        {"xmin":0.0475,"ymin":0.5127118644067796,"xmax":0.3175,"ymax":0.9364406779661016}
        ],
    "visited":1,
    "specie":"Amegilla quadrifasciata"}  
    nb : the case where there are bees from several species on the same image is not considered
    '''

    #read json file
    with open(json_path) as json_file:
        data = json.load(json_file)

    #create output directory
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #create subfolder for specie
    specie = data[0]['specie']
    #base output path
    base_output_path = os.path.join(output_path,specie)


    if not os.path.exists(os.path.join(output_path,specie)):
        os.mkdir(os.path.join(output_path,specie))

    count_errors = 0

    #crop images
    for data_img in data:

        #get image path
        img_path = data_img['file_path']

        # replace path to image to point to bd_71 folder
        img_path = img_path.replace('./BD_71',bd71_path)


        #get image name and extension
        img_name = img_path.split('/')[-1].split('.')
        img_name = '-'.join(img_name[:-1])
        img_extension = img_path.split('/')[-1].split('.')[-1]

        #get image size
        try :
            img = cv2.imread(img_path)
        except:
            print('Image {} not found'.format(img_path))
            count_errors +=1
            print('Number of errors : {}'.format(count_errors))
            continue

        #get image size
        try:
            h,w,_ = img.shape
        except:
            continue

        #get boxes
        boxes = data_img['boxes']

        for box in boxes:

            #crop image
            try :
                img2 = crop_image(img,box,w,h)
            except:
                print('Error while cropping image {}'.format(img_path))
                print('Image {} not found'.format(img_path))
                count_errors +=1
                print('Number of errors : {}'.format(count_errors))
                continue

            #save image

            # case where image already exists (several bees on the same image)
            if os.path.exists(os.path.join(base_output_path,img_name + '.' + img_extension)):
                ind_aux = 1
                final_path = os.path.join(base_output_path,img_name + '-' + str(ind_aux) + '.' + img_extension)

                while os.path.exists(os.path.join(base_output_path,img_name + '-' + str(ind_aux) + '.' + img_extension)):
                    ind_aux +=1 
                    final_path = os.path.join(base_output_path,img_name + '-' + str(ind_aux) + '.' + img_extension)

            else:
                final_path = os.path.join(base_output_path,img_name + '.' + img_extension)  

            try:
                img2.save(final_path)
            except:

                continue
 
    return count_errors,specie

def crop_from_jsons(jsons_path,output_path,bd71_path):
    '''
    Crop images from all json files in a folder
    '''

    count_errors_dict  = {}

    for json_path in os.listdir(jsons_path):
        count_error,specie = crop_from_json(os.path.join(jsons_path,json_path),output_path,bd71_path)
        count_errors_dict[specie] = count_error


    


if __name__ == '__main__':
    args = argparser.parse_args()
    jsons_path = args.jsons_path
    output_path = args.output_path
    bd71_path = args.bd71
    crop_from_jsons(jsons_path,output_path,bd71_path)