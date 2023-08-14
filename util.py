import pickle

from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))

#This function checks if the spot is empty or not 
# #Takes an image as an output and reshapes it
def empty_or_not(spot_bgr):

    flat_data = []                                  #List of reshaped image data

    img_resized = resize(spot_bgr, (15, 15, 3))     #reshaping the image
    flat_data.append(img_resized.flatten())         #inserts the reshaped image in the list flat_data
    flat_data = np.array(flat_data)                 #Converting the image data to numpy array

    y_output = MODEL.predict(flat_data)             #Inserting the image data into the trained model

    #Outputs a boolean value YES OR NOT
    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY

#This function extracts the bounding box from the mask_crom.png
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)    #Axis X1
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)     #Axis Y1
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)    #Width
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)   #Height

        slots.append([x1, y1, w, h])

    return slots

