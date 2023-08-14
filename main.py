import cv2

from util import get_parking_spots_bboxes   #From the util.py file importing the functions get_parking_spots_bboxes
from util import empty_or_not               #importing empty_or_not func
import numpy as np
from matplotlib import pyplot as plt

#this function substract the mean value of first image pixels with second image pixels
#The output shows how similar two images are or how different two images are from each other
#Here images are frames of the video
def calc_diffs(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


# creating mask png path variable 
mask = './mask_1920_1080.png'

# Creating video path variable with data path
video_path = './data/parking_1920_1080_loop.mp4'

#Reading mask from memory
mask = cv2.imread(mask, 0)

#mask_png Bounding box function
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

#calling get_parking_spots_bboxes from utils
spots = get_parking_spots_bboxes(connected_components)

# print(spots[0])

spots_status = [None for j in spots]        #A list of how many parkings spots are empty and filled
diffs = [None for j in spots]               #A List 

previous_frame = None                       #This variable is to store previous frame image, 
                                            #We will use this to compare each frames with the previous frames

# Calling VideoCapture Function from OpenCV
cap = cv2.VideoCapture(video_path)

ret = True

step = 30       #How ofen we are going to classify our parking spots


frame_nmr = 0


#Iterating all the frames through the video to visualize it
while ret:
    ret, frame = cap.read()
    
    #This condition loops through each spotsw in the current image frame and compares it with the previous image frame
    #Calling the calc_diffs function
    if frame_nmr % step ==  0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            
            diffs[spot_indx] = calc_diffs(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        
        #These commented print shows the numpy iteration of each images
        #print(diffs)
        #print([diffs[j] for j in np.argsort(diffs)][::-1])
        #SHOWS HISTOGRAMS
        # plt.hist([diffs[j]/np.amax(diffs) for j in np.argsort(diffs)][::-1])
        
        # if frame_nmr == 300:
        #     plt.show()
        
        
    #This condition updates the spot_status count every 30 frames   
    if frame_nmr % step ==  0:
        #for spot_indx, spot in enumerate(spots):
        
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs)>0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            
            #Spot contains only empty parking spots
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            #Calling the function passing spot_crop in the parameters
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status    
    
    #Saving the value of previous frame in to previous_frame variable
    if frame_nmr % step ==  0:
        previous_frame = frame.copy()
    
       
    #This loop iterates throu the frames and draws bounding boxes
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]
        
        #Adjusting the coloring of empty parking spot and Initiating Green or Red Bounding box around parking spots
        if spot_status:         #if the spot is empty then it will show green box around them
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)          
        else:                   #if the spot is not empty then it will show Red box around them
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
    
    
    cv2.rectangle(frame, (80, 20), (630, 110), (0,0,0), -1)  
    #Counter Text for number of empty spots
    cv2.putText(frame, "Available Empty Spots: {}/{}".format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.putText(frame, "Press Q to exit", (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
       
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)   #Showing the frames
    
    if cv2.waitKey(25) & 0xFF == ord('q'):  #If the letter 'q' is pressed then the windows will be closed
        break
    
    
    frame_nmr += 1
# releasing memory for the object
cap.release()
cv2.destroyAllWindows()     #Closing all windows

