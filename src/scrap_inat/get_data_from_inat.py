import os
import wget
from tqdm import tqdm
import asyncio
import argparse

'''
See some documentation here:
https://github.com/inaturalist/inaturalist-open-data/tree/documentation
'''



argparser = argparse.ArgumentParser()


argparser.add_argument(
    '-i', '--input',
    help="""Csv of metadata from inat.db with all the pictures to download. 
	 		-Structure :
			#taxon_name, #taxon_id, #photo_id, #extension""",
    required=True)

args = argparser.add_argument(
    '-o', '--output',
    help='Output folder, where images will be downloaded',
    required=True)

args = argparser.add_argument(
    '-s', '--size',
	help="""
    Size of the images to download.

	Choose between:

		original - 2048px
		large - 1024px
		medium - 500px
		small - 240px
		thumb - 100px
		square - exactly 75x75px, cropped to be square
				
		""",
	required=True,
	choices=['original', 'large', 'medium', 'small', 'thumb', 'square'])

# use this decorator to run a function in a background thread
def background(f):
	def wrapped(*args, **kwargs):
		return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
		
	return wrapped

# download image in background thread from url 
@background
def get_image(image_url, target_dest):
	wget.download(image_url, target_dest)
	return

# main function to download images
def download_images(input_file, output_folder,size):
	''''
	DOWNLOAD IMAGES FROM INATURALIST

	Image will be downloaded in the following folder structure:
		- output_folder
			- taxon_name_1
				- photo_id.extension
				- ...
			- taxon_name_2
			- ...
	'''


	# Create output folder
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	# Load CSV of selected pictures : #taxon_id	#photo_id #extension
	with open(input_file, newline='') as csvfile:
		lines = csvfile.read().split("\n")
		for i,row in enumerate(tqdm(lines)):
			data = row.split(',')
			if i > 0 and len(data) > 2:
				taxon_name = data[0]
				photo_id = data[2]
				extension = data[3]

				# sometimes taxon name may be stored like this :
				# "apis mellifera" rather than apis mellifera
				if taxon_name[0] == '"':
					taxon_name = taxon_name[1:-1]
			
				if not os.path.exists(os.path.join(output_folder, taxon_name)):
					os.mkdir(os.path.join(output_folder, taxon_name))
					
				image_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{extension}"
				target_dest = os.path.join(output_folder, taxon_name, f"{photo_id}.{extension}")
				get_image(image_url, target_dest)


if __name__ == "__main__":
	args = argparser.parse_args()
	download_images(args.input, args.output, args.size)