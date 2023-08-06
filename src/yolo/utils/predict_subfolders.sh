#!/bin/bash

# define the input folder
INPUT_FOLDER="../data_bees_detection/whole_dataset/inat_25_04"

# loop through all subdirectories within the input folder
for dir in "$INPUT_FOLDER"/*/; do
  # check that the subdirectory contains files
  if [ -n "$(ls -A $dir)" ]; then
    # run the predict.py script with the --i argument set to the subdirectory path
    python3 /src/yolo/predict.py --i "$dir"
  fi
done

python3 '/workspaces/projet_bees_detection_basile/merge_csv.py'

python3 /workspaces/projet_bees_detection_basile/temp.py

python3 /src/crop/crop_from_csv.py
