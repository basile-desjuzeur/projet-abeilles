import pandas as pd
from IPython.display import display
#paths to data

videos_annotated = "/src/data/all_videos_annotated.csv"

#on veut changer le csv en dataframe

df_videos = pd.read_csv(videos_annotated, sep=",")
display(df_videos)

#on compte les apparaition de chaque espèce

labels =["Anthophila"]
count = {label:0 for label in labels}

for specie in df_videos['species1']:
    for label in labels:
        if label in specie:
            count[label] += 1

for specie in df_videos['species2']:
    print(specie)
    print(type(specie))
    print(type("nan"))
    print(specie != "nan")
    if str(specie) != "nan":
        for label in labels:
            if label in specie:
                count[label] += 1
            
for specie in df_videos['species3']:
  if str(specie) != "nan":
        for label in labels:
            if label in specie:
                count[label] += 1

for specie in df_videos['species4']:
 if str(specie) != "nan":
        for label in labels:
            if label in specie:
                count[label] += 1
print(count)