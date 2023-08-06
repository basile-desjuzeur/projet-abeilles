import os 
import pandas as pd
import re 
from tqdm import tqdm

path_1 = '/home/basile/Documents/projet_bees_detection_basile/data_bees_detection/whole_dataset/inat_25_04'
path_2 = '/home/basile/Documents/projet_bees_detection_basile/data_bees_detection/whole_dataset/iNaturalist'


# Get the paths of the files in path_1 in a dataframe

paths_1 = []

for folder in tqdm(os.listdir(path_1)):

    for file in os.listdir(os.path.join(path_1,folder)): 

        path = os.path.join(path_1,folder,file)

        paths_1.append(path)

df_path_1 = pd.DataFrame(paths_1, columns=['path_1'])

# Get the numbers of the files in path_1 in a column number

df_path_1['number'] = df_path_1['path_1'].apply(lambda x : x.split('/')[-1].split('.')[0])

# Get the paths of the files in path_2 in a dataframe

paths_2 = []

for folder in tqdm(os.listdir(path_2)):

    for file in os.listdir(os.path.join(path_2,folder)): 

        path = os.path.join(path_2,folder,file)

        paths_2.append(path)

df_path_2 = pd.DataFrame(paths_2, columns=['path_2'])

# Get the numbers of the files in path_2 in a column number

df_path_2['number'] = df_path_2['path_2'].apply(lambda x : x.split('/')[-1].split('.')[0])
df_path_2['number'] = df_path_2['number'].apply(lambda x: re.findall(r'\d+', x)[0])


# Get the numbers that are in path_1 and in path_2
    
df = pd.merge(df_path_1, df_path_2, how='inner', on='number')

# Get the paths of the files in path_1 and path_2 in a dataframe

df_paths = pd.DataFrame(df['path_1'])
df_paths['path_2'] = df['path_2']


# Get the name of the specie 

df_paths['specie_path_1'] = df_paths['path_1'].apply(lambda x : x.split('/')[-2])
df_paths['specie_path_2'] = df_paths['path_2'].apply(lambda x : x.split('/')[-1])
# Stored as Bombus pascuorum85957.jpeg wanna get Bombus pascuorum8597
df_paths['specie_path_2'] = df_paths['specie_path_2'].apply(lambda x : x.split('.')[0])
# Stored as Bombus pascuorum8597 wanna get Bombus pascuorum using the regex
df_paths['specie_path_2'] = df_paths['specie_path_2'].apply(lambda x: re.sub(r"\d+", "", x))


# Check if the specie in path_1 is the same as the specie in path_2

df_paths['same_specie'] = df_paths['specie_path_1'] == df_paths['specie_path_2']

# Keep only the files that have the same specie in path_1 and path_2

df_paths = df_paths[df_paths['same_specie'] == True]

df_paths.to_csv('paths.csv', index=False)