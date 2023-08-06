# Train config
python3 train.py -c config/bees_detection.json

# Evaluate config
python3 evaluate.py -c config/bees_detection.json 

# Real time
python3 predict.py -c config/bees_detection.json -r True -i 0

# Image.s/Video prediction
python3 predict.py -c config/bees_detection.json -i <path to file/directory>

# Multi training
sh multi_train.sh <path to file that list config files>
