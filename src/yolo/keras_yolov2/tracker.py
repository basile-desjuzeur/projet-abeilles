import numpy as np
from collections import OrderedDict


class BoxTracker():
  def __init__(self, maxDisappeared=20):
    self.nextObjectID = 0
    self.objects = OrderedDict() #dictionnaire qui se rappelle de l'ordre d'insertion
    self.disappeared = OrderedDict() #dictionnaire qui se rappelle de l'ordre d'insertion
    self.maxDisappeared = maxDisappeared
    self.tracker_history = {}

  def register(self, box): #on enregistre une nouvelle boîte d'un nouvel oiseau
    """
    Register a new box to the current boxes.
    """
    box.id = self.nextObjectID
    self.objects[self.nextObjectID] = box 
    self.disappeared[self.nextObjectID] = 0 
    self.nextObjectID += 1
  
  def deregister(self, objectID): #on enlève les box des oiseaux qui sont partis
    """
    Remove a box from the know boxes.
    """
    del self.objects[objectID]
    del self.disappeared[objectID]
  
  def compute(self, inputBoxes):  #important car choisit les boîtes qui correspondent au déplacement d'un même oiseau ou à l'arrivée d'un nouvel oiseau
    """
    Compute next iteration track.
    """
   
    ### Pas de nouvelle boîte
    if len(inputBoxes) == 0:
      for objectID in list(self.disappeared.keys()):
        # Update TTL
        self.disappeared[objectID] += 1

        # Remove boxes that excedeed maxTTL
        if self.disappeared[objectID] > self.maxDisappeared:
          self.deregister(objectID)
      
      # Nothing more to do
      return self.objects

    ### No previous box
    if len(self.objects) == 0:
      # Add every input box
      for i in range(0, len(inputBoxes)):
        self.register(inputBoxes[i])
      
      # Nothing more to do
      return self.objects
    
    ### Update previous boxes with new ones

    # Get ids and boxes
    objectIDs = list(self.objects.keys())#liste des id des boîtes précédentes
    boxes = list(self.objects.values())#liste des boîtes précédentes

    # Compute distances pour aplliquer le iou score

    D = distances_boxes(boxes, inputBoxes) 
    # i id dans la frame = D possède i lignes et j boites englobantes dans la frame = D possède j colonnes
    #Si 2 id et 2 boxes alors on a une matrice 2x2 comme: [[0.05105336 0.00728146] [0.0603838  0.07381677]]
    #Si 1 id et 2 boxe (imbriquées) dans la frame alors on a une matrice de distance de taille 1*2: [[0.00728146 0.05105336]]
    #Si 1 id et 1 boxe dans la frame alors on a une matrice de distance de taille 1*1: [[0.05105336]]
   
    
    # Get min distances per previous and new boxes

    rows = D.min(axis=1).argsort() #renvoie les indices des lignes triées par ordre croissant des distances
  
    cols = D.argmin(axis=1)[rows] #renvoie les indices des colonnes qui correspondent aux distances les plus petites
    

    usedRows = set()
    usedCols = set()
    for (row, col) in zip(rows, cols):
      # Already used, go next
      if row in usedRows or col in usedCols: #si la ligne ou la colonne a déjà été utilisée, on passe à la suivante
        continue

      # Change previous box to new one
     
      objectID = objectIDs[row] #objectID est l'id de la boîte précédente sous forme de liste et objectIDs[row] est l'id de la boîte précédente sous forme de nombre
      self.objects[objectID] = inputBoxes[col]
      self.objects[objectID].id = objectID
      self.disappeared[objectID] = 0

      # Set current row and col to used
      usedRows.add(row)
      usedCols.add(col)
    
    # Compute unused rows and cols
    unusedRows = set(range(0, D.shape[0])).difference(usedRows) #renvoie les indices des lignes qui n'ont pas été utilisées
    unusedCols = set(range(0, D.shape[1])).difference(usedCols) #renvoie les indices des colonnes qui n'ont pas été utilisées
    
    # Update boxes TTL
    if D.shape[0] >= D.shape[1]: #si le nombre de boîtes englobantes est supérieur au nombre d'oiseaux (ex: 2 boîtes englobantes et 1 oiseau dans 2022-08-22-09-57-28_C1_1MESBLE-1SITTOR.mp4 )
      for row in unusedRows:
        objectID = objectIDs[row]
        self.disappeared[objectID] += 1

        # Kill boxes that excedeed maxTTL
        if self.disappeared[objectID] > self.maxDisappeared: #si le nombre de frames où une boîte englobante ne bouge pas est supérieur à maxDisappeared, on enlève la boîte englobante
          self.deregister(objectID)

    # Create new track
    else:
      for col in unusedCols:
        self.register(inputBoxes[col])
      
    return self.objects

  def update(self, inputBoxes):
    """
    Update boxtracker with next iteration.
    """
    boxes = self.compute(inputBoxes)
    
    # Save hisotry in dict

    for box in boxes.values():
      if box.id in self.tracker_history:
        self.tracker_history[box.id].append((box.get_label(), box.get_score()))
      else:
        self.tracker_history[box.id] = [(box.get_label(), box.get_score())]

    return boxes


def distance_box(box1, box2):
  """
  Compute distance bewteen two boxes.
  """
  cx1, cy1 = (box1.xmin + box1.xmax) / 2, (box1.ymin + box1.ymax) / 2
  cx2, cy2 = (box2.xmin + box2.xmax) / 2, (box2.ymin + box2.ymax) / 2

  # Euclidian distance between centroids
  return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def distances_boxes(boxes1, boxes2, dist=distance_box):
  """
  Compute distance matrix between two sets of boxes.
  """

  res = np.zeros((len(boxes1), len(boxes2)))

  # Compute distances 2 by 2
  for i, box1 in enumerate(boxes1):
    for j, box2 in enumerate(boxes2):
      res[i, j] = dist(box1, box2)
  return res

def NMS(inputBoxes, overlapThresh=0.4): #overlapThresh est le seuil de recouvrement à partir duquel on considère qu'il y a 2 boîtes englobantes pour un même oiseau
  """
  -- Non Maximal Supression --
  Remove boxes that overlap and which are not the best ones.
  """
  
  # Return an empty list, if no boxes given
  if len(inputBoxes) == 0:
    return []
  
  # Convert boxes to (:,4) array
  boxes = np.zeros((len(inputBoxes), 4))
  for i, box in enumerate(inputBoxes):
    boxes[i] = np.array([box.xmin, box.ymin, box.xmax, box.ymax])
  
  x1 = boxes[:, 0]  # x coordinate of the top-left corner
  y1 = boxes[:, 1]  # y coordinate of the top-left corner
  x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
  y2 = boxes[:, 3]  # y coordinate of the bottom-right corner

  # Compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  areas = (x2 - x1 + 1) * (y2 - y1 + 1) #on obtient un vecteur contenant les aires des boîtes englobantes de l'ensemble inputBoxes

  # The indices of all boxes at start. We will redundant indices one by one.
  indices = np.arange(len(x1))
  for i,box in enumerate(boxes):
    # Create temporary indices  
    temp_indices = indices[indices != i]

    # Find out the coordinates of the intersection box
    xx1 = np.maximum(box[0], boxes[temp_indices, 0])
    yy1 = np.maximum(box[1], boxes[temp_indices, 1])
    xx2 = np.minimum(box[2], boxes[temp_indices, 2])
    yy2 = np.minimum(box[3], boxes[temp_indices, 3])

    # Find out the width and the height of the intersection box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # Compute the ratio of overlap
    overlap = (w * h) / areas[temp_indices]

    # If the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
    if np.any(overlap) > overlapThresh:
      indices = indices[indices != i]

  # Return only the boxes at the remaining indices
  outputBoxes = []
  for i in indices:
    outputBoxes.append(inputBoxes[i])
  return outputBoxes