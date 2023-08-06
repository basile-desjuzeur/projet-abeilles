import os
from xml.dom import minidom
import csv  
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import wget
import pickle
import random
from PIL import Image
import json
from operator import itemgetter


source_path = 'AnnotationAbeilles/BD_71/'
target_path = 'Cropped_BD_71/'
target_cap_path = 'BD_71/'

with open('liste_classes_71.csv', newline='') as csvfile:
	filereader = csv.DictReader(csvfile, delimiter=',')
	for row in filereader:
		specie = row['species']
		print(specie)

		# Opening JSON file
		f = open('json_checked/' + specie + '.json')
		  
		# returns JSON object as 
		# a dictionary
		data = json.load(f)

		if not os.path.exists(target_path + specie):
			os.mkdir(target_path + specie)
		if not os.path.exists(target_cap_path + specie):
			os.mkdir(target_cap_path + specie)

		for img in data:
		
			if img['visited']==1:
			
				img_path = img['file_path'][8:]
				
				aux = img_path.split('.')
				img_extension = aux[-1]
				img_name = img_path[:-len(img_extension)]
				
				
				boxes = img['boxes']
				
				img = Image.open(source_path + img_path)
				w, h = img.size
				
				for box in boxes:
					xmin = min(int(box['xmin']*w), int(box['xmax']*w))
					xmax = max(int(box['xmin']*w), int(box['xmax']*w))
					ymin = min(int(box['ymin']*h), int(box['ymax']*h))
					ymax = max(int(box['ymin']*h), int(box['ymax']*h))
					
					print(img_name, img_extension, xmin, xmax, ymin, ymax)
					if xmin==xmax or ymin==ymax:
						continue
					else:
						img2 = img.crop((xmin, ymin, xmax, ymax))
						if os.path.exists(target_path + img_path):
							ind_aux = 1
							while os.path.exists(target_path + img_name + '-' + str(ind_aux) + '.' + img_extension):
								ind_aux +=1
							img2.save(target_path + img_name + '-' + str(ind_aux) + '.' + img_extension) 
						else:
							img2.save(target_path + img_path) 
							
						img.save(target_cap_path + img_path)
					

		f.close()

