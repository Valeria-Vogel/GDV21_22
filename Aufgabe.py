
import cv2
import numpy as np
import copy

# declare variables
width = 2500
height = 1500
gradient_color = 0
pixel = 15

# create black image
img = np.zeros((height,width,3), np.uint8)

# turn black image into gradient image using steps of 10 pixels for each 1 of the 255 rgb range
for i in range (width):
    if gradient_color < 255 and i % 10 == 0:
        gradient_color += 1
    for j in range (height):
        img[j,i] = (gradient_color, gradient_color, gradient_color)

# create middle square with deepcopy, so that it isn't referencing the live video
middle = copy.deepcopy(img[625:875,1125:1375])

# copy middle square to the top left and top right
img[20:270,20:270] = middle
img[20:270, 2230:2480] = middle

# deepcopy the current image, so that it can be used to restore the background in the live video
imgcopy = copy.deepcopy(img)

# title and window for the video
title = 'Gradient'
cv2.namedWindow(title,  cv2.WINDOW_FREERATIO) 

# create loop
while True:

    # use loop to move the square 3 pixels to the right for every iteration
    if pixel < 2232:
        pixel += 3
        if (pixel == 2232):
            pixel = 21

    # copy middle square onto squares in the image, using the pixel loop
    img[625:875, pixel:pixel+250] = middle

    #set other parts of the image back to what color they used to be
    img[0:height, 0:pixel-1] = imgcopy[0:height, 0:pixel-1]
    img[0:height, pixel+251:width] = imgcopy[0:height, pixel+251:width]


    # display the image
    cv2.imshow(title, img)

    # press q to get out of the loop
    if cv2.waitKey(10) == ord('q'): 
        break
# close window
cv2.destroyAllWindows()