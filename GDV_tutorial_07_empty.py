import cv2
import numpy as np

### Goal: Count the number of green smarties in the images
# define green in HSV
hue = 60  # 60 is pure green // 0 - 180 in OpenCV
hue_range = 10
saturation = 155
saturation_range = 100
value = 155
value_range = 100
lower_green = np.array([hue - hue_range,saturation - saturation_range,value - value_range])
upper_green = np.array([hue + hue_range,saturation + saturation_range,value + value_range])

# load image
img = cv2.imread('images\smarties03.JPG',cv2.IMREAD_COLOR)
#img = cv2.resize(img,(800,600))

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Umwandlung von bgr in hsv

# create a mask
mask = cv2.inRange(hsv, lower_green, upper_green) #damit man nur die grünen Teile sieht

## morphological operations code
# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

# dilation with parameters
def dilatation(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.dilate(img, element)

# erosion with parameters
def erosion(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.erode(img, element)

# opening
def opening(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
# opening
def closing(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)





#cv2.imshow('Masked Image', mask)
#cv2.waitKey(0)

#mask = dilatation(mask,kernel_size,kernel_shape) (see https://docs.opencv.org/3.4.15/db/df6/tutorial_erosion_dilatation.html)
kernel_size = 7 
kernel_shape = morph_shape(2)
mask = opening(mask, kernel_size, kernel_shape)
mask = closing(mask, kernel_size, kernel_shape)
# mask out green parts as binary image

# find connected components
connectivity = 8
(numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)


# find center of mass and draw a mark in the original image
red_BGR = (0,0,255)
circle_size = 10
circle_thickness = 5
min_size = 10
numRejected = 1


# go through all (reasonable) found connected components
for i in range(1,numLabels): #schleife startet mit 1 kann in einigem Code FALSCH sein
    # check size and roundness as plausibility
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    if w < min_size or h < min_size:
        print('Found a too small component.')
        numRejected += 1
        continue # found component is too small to be correct 
    if w > h:
        roundness = 1.0 / (w/h)
    elif h > w:
        roundness = 1.0 / (h/w)  
    if (roundness < .9):
        print ('Found a component that is not round enough.')
        numRejected += 1
        continue # ratio of width and height is not suitable

    # find and draw center
    center = centroids[i]
    center = np.round(center)
    center = center.astype(int)
    cv2.circle(img,center,circle_size,red_BGR,circle_thickness)
    
    # find and draw bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) 

# print out number of connected components

print('We have found', str(numLabels - numRejected), 'green smarties.') #numberd Labels 


# show the original image with drawings in one window
cv2.imshow('Original image', img)
# show the masked image in another window

# show the mask image in another window
cv2.imshow('Masked Image', mask)
cv2.imwrite('mask.jpg', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
