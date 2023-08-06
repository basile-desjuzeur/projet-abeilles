path_to_images="../data_bees_detection/BD_71"

for folder in $(ls $path_to_images)
do
    for file in $(ls $path_to_images/$folder)
    do
        echo "$file" >> "/src/data/all_videos.csv"
        
    done
done

