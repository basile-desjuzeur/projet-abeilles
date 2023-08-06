#importations
import csv

#paths 
video_path = "/src/data/all_videos.csv"
annotated_video_path = "/src/data/annotated_videos.csv"


###Nos listes de labels connus ou non

#noms des labels que l'on a dans toutes les vidéos (ou que l'on pense avoir jusqu'à confirmation de maxime)
labels = ["Anthophila"]
#noms des labels que notre train contient
labels_du_train =["Anthophila"]

###on veut extraire le nom des oiseaux passant sur chaque vidéos et écrire des listes contenant le nom de la vidéos et le nom des oiseaux qui passent sur cette vidéo tout en respectant l'ordre de passage

with open(video_path, 'r') as annotated_video_file:
    reader = csv.reader(annotated_video_file)
    #on crée une liste contenant nom de la vidéo, nom du 1er oiseau, nom du 2ème oiseau etc
    L = []
    for line in annotated_video_file:
        line = line[:-1] if line[-1] == '\n' else line
        #on crée une liste contenant les éléments de la ligne séparés par _
        L.append(line.split("_"))

#on crée une liste contenant les noms des vidéos   
with open(video_path, 'r') as annotated_video_file:
    reader = csv.reader(annotated_video_file)
    #on crée une liste contenant nom de la vidéo, nom du 1er oiseau, nom du 2ème oiseau etc
    nom_video = []
    for line in annotated_video_file:
        line = line[:-1] if line[-1] == '\n' else line
        #on crée une liste contenant les éléments de la ligne séparés par _
        nom_video.append(line)



final_list = []
for i in range(len(L)):
    #on crée une liste contenant le nom de la vidéo et le nom de chaque oiseau

    #on détache le .mp4 du nom de la vidéo
    line = L[i][-1].split(".")
    line.pop()
    L[i].append(line[0])  
    #on contruit une nouvelle liste ne contenat que le nom de la vidéo et le nom de chaque oiseau associé à leur nombre d'occurences
    #Attention ici on ne récupère que les oiseaux présent dans le train et les roufam que l'on renommera après en rougor car erreur d'annotation des ornitho
    #Pour les autres espèces il faudra aggrandir la liste des labels

    new_list = [nom_video[i]]
    for j in range(len(L[i])):
        for w in range(len(labels)):
            if labels[w] in L[i][j]:  
                new_list.append(L[i][j])
    final_list.append(new_list)


#on enlève les .mp4 qui restent

for i in range(len(final_list)):
    for j in range(1,len(final_list[i])-1): #on ne veut pas retirer le nom de la vidéo donc on commence  à 1
        if '.mp4' in final_list[i][j]: 
            del final_list[i][j]

# on s'oocupe des cas particuliers liés à l'erreur d'annotation des ornitho

#on remplace les roufam par les rougor

for i in range(len(final_list)):
    for j in range(1,len(final_list[i])):
        if '1ROUFAM' in final_list[i][j]:
            final_list[i][j] = '1ROUGOR'
        if '2ROUFAM' in final_list[i][j]:
            final_list[i][j] = '2ROUGOR'
        if '3ROUFAM' in final_list[i][j]:
            final_list[i][j] = '3ROUGOR'
        if '4ROUFAM' in final_list[i][j]:
            final_list[i][j] = '4ROUGOR' #on suppose qu'il n'y a pas plus de 4 roufam sur une vidéo

#on remplace les mesbleu par les mesble 

for i in range(len(final_list)):
    for j in range(1,len(final_list[i])):
        if '1MESBLEU' in final_list[i][j]:
            final_list[i][j] = '1MESBLE'
        if '2MESBLEU' in final_list[i][j]:
            final_list[i][j] = '2MESBLE'
        if '3MESBLEU' in final_list[i][j]:
            final_list[i][j] = '3MESBLE'
        if '4MESBLEU' in final_list[i][j]:
            final_list[i][j] = '4MESBLE' #on suppose qu'il n'y a pas plus de 4 mesbleu sur une vidéo

#on remplace les siteur par les sittor (si cela est la même espèce)

for i in range(len(final_list)):
    for j in range(1,len(final_list[i])):
        if '1SITEUR' in final_list[i][j]:
            final_list[i][j] = '1SITTOR'
        if '2SITEUR' in final_list[i][j]:
            final_list[i][j] = '2SITTOR'
        if '3SITEUR' in final_list[i][j]:
            final_list[i][j] = '3SITTOR'
        if '4SITEUR' in final_list[i][j]:
            final_list[i][j] = '4SITTOR' #on suppose qu'il n'y a pas plus de 4 siteur sur une vidéo

#print(final_list)
print("Longueur de la final_list avant suppression des labels inconnu du train", len(final_list))
print("\n")

'''Remarque sur le dataset à ce niveau:
- on a des images qui sont mal annotées avec une barre - à la place d'une barre _ donc le parsing ne marche pas
-on n'est pas sûr si certaines nouvelles annotations sont de nouvelles espèces ou une autre annotation pour une autre espèces: sittor et siteur (on a choisi avec Axel de dire que c'était les mêmes)
-certaines images ne seront associées à aucune espèces car leur espèce ne sont pas encore dans les labels connus
-on ne peut pas garder une vidéo avec une espèces encore absente du train car même si elle contient d'autres espèces connues, on ne peut pas prédire l'espèce absente'''

#on retire les vidéos avec au moins une espèce absente du train, c'est-à-dire ceux dont un élément dans la liste se trouve dans labels mais pas dans label_du_train

suppr = []
for i in range(len(final_list)):
    for j in range(1,len(final_list[i])):
        for label in labels:
            if label in final_list[i][j] and label not in labels_du_train:
                suppr.append(final_list[i])
            
       

print("Liste des images supprimée car possédant au moins un oiseau non dans les labels:",suppr)
print("Nombre d'images supprimées car possédant au moins un oiseau non dans les labels:",len(suppr))
print("\n")

#on élimine les listes de final_list qui sont dans suppr

for i in range(len(suppr)):
    final_list.remove(suppr[i])

# print("Final_listtemporaire:",final_list)
# print("Longueur de la final_list après suppression des labels inconnu du train", len(final_list))
# print("\n")

#on remplace les '1+nom_du_label' par seulement nom du label 
#  Pour i supérieur ou égal à 2 on remplace les 'i + 'nom_du_label' par i élément que l'on rajoute à la liste de final_list[i][0]

for i in range(len(final_list)):
    for j in range(1,len(final_list[i])):
        if '1' in final_list[i][j]:
            final_list[i][j] = final_list[i][j][1:]
        for z in range(2,10):
            if str(z) in final_list[i][j]:
                final_list[i][j] = final_list[i][j][1:]
                for w in range(z-1): #on le rajoute autant de fois -1 (car on l'a déjà rajouté une fois en haut en le renommant) que le nombre indiqué dans le nom de l'espèce
                    #on souhaite insérer l'élément à la position j+1
                    final_list[i].insert(j+1,final_list[i][j]) 

print("Final_list après la mofification des 1 et des 2,3... en tenant compte de l'ordre d'arrivée:", final_list)
print("Longueur de la final_list après suppression des labels inconnu du train", len(final_list))
print("\n")
print("test du respect de l'ordre dans le cas d'un mesnon mescha mescha:" ,final_list[82],"l'ordre des éléments de la liste est bien respecté")

#on écrit final_list dans un csv appelé all_videos_annotated.csv

with open('all_videos_annotated.csv', 'w', newline='') as file:
    writer = csv.writer(file)   
    writer.writerows(final_list)
    

