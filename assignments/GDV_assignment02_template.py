'''
Assignement 02: Object counting
Group: 03
Names: Valeria Vogel, Maximilian Flack
Date: 15.11.2021
Sources: Code from GDV Tutorial 07, https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html, https://colorpicker.me/#04b00c
'''

import cv2
import numpy as np
import glob # for loading all images from a directory

### Goal: Count the number of all colored balls in the images

# ground truth
num_yellow = 30
num_blue = 5
num_pink = 8
num_white = 10
num_green = 2
num_red = 6
gt_list = (num_red, num_green, num_blue, num_yellow, num_white, num_pink)

# define color ranges in HSV, note that OpenCV uses the following ranges H: 0-179, S: 0-255, V: 0-255 

# red
hue_red = [5,174]
hue_range_red = 5
saturation_red = 200
saturation_range_red = 55
value_red = 155
value_range_red = 100
# two ranges for red, cause red balls are in ranges 0-10 and 169-179 
bottom_lower_red = np.array([hue_red[0] - hue_range_red,saturation_red - saturation_range_red,value_red - value_range_red])
bottom_upper_red = np.array([hue_red[0] + hue_range_red,saturation_red + saturation_range_red,value_red + value_range_red])
top_lower_red = np.array([hue_red[1] - hue_range_red,saturation_red - saturation_range_red,value_red - value_range_red])
top_upper_red = np.array([hue_red[1] + hue_range_red,saturation_red + saturation_range_red,value_red + value_range_red])

# green
hue_green = 60
hue_range_green = 20
saturation_green = 155
saturation_range_green = 100
value_green = 155
value_range_green = 100
lower_green = np.array([hue_green - hue_range_green,saturation_green - saturation_range_green,value_green - value_range_green])
upper_green = np.array([hue_green + hue_range_green,saturation_green + saturation_range_green,value_green + value_range_green])

# blue
hue_blue = 100
hue_range_blue = 20
saturation_blue = 155
saturation_range_blue = 100
value_blue = 155
value_range_blue = 100
lower_blue = np.array([hue_blue - hue_range_blue,saturation_blue - saturation_range_blue,value_blue - value_range_blue])
upper_blue = np.array([hue_blue + hue_range_blue,saturation_blue + saturation_range_blue,value_blue + value_range_blue])

# yellow
hue_yellow = 20
hue_range_yellow = 15
saturation_yellow = 180
saturation_range_yellow = 75
value_yellow = 207
value_range_yellow = 48
lower_yellow = np.array([hue_yellow - hue_range_yellow,saturation_yellow - saturation_range_yellow,value_yellow - value_range_yellow])
upper_yellow = np.array([hue_yellow + hue_range_yellow,saturation_yellow + saturation_range_yellow,value_yellow + value_range_yellow])

# white
hue_white = 50
hue_range_white = 40
saturation_white = 95
saturation_range_white = 95
value_white = 205
value_range_white = 50
lower_white = np.array([hue_white - hue_range_white,saturation_white - saturation_range_white,value_white - value_range_white])
upper_white = np.array([hue_white + hue_range_white,saturation_white + saturation_range_white,value_white + value_range_white])

# pink
hue_pink = 10
hue_range_pink = 10
saturation_pink = 100
saturation_range_pink = 70
value_pink = 205
value_range_pink = 50
lower_pink = np.array([hue_pink - hue_range_pink,saturation_pink - saturation_range_pink,value_pink - value_range_pink])
upper_pink = np.array([hue_pink + hue_range_pink,saturation_pink + saturation_range_pink,value_pink + value_range_pink])


### morphological operations
# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

# dilation with parameters
def dilation(img,size,shape,iterations): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.dilate(img, element, iterations=iterations)

# erosion with parameters
def erosion(img,size,shape,iterations): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.erode(img, element, iterations=iterations)

# opening
def opening(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, element)

# closing
def closing(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)


# set color under test
num_colors = 6
color_names = ['red', 'green', 'blue', 'yellow', 'white','pink']

# set ranges for masks
color_ranges = [bottom_upper_red, bottom_lower_red, upper_green, lower_green, upper_blue, lower_blue, upper_yellow, lower_yellow, upper_white, lower_white, upper_pink, lower_pink]

# setting the parameters that work for all colors

# set individual (per color) parameters

# parameters for testing purposes
red_BGR = (0,0,255)
green_BGR = (0,255,0)
circle_size = 10
circle_thickness = 5
min_size = 10


num_test_images_succeeded = 0

for img_name in glob.glob('images/chewing_gum_balls*.jpg'): 
    # load image
    print ('Searching for colored balls in image:',img_name)

    all_colors_correct = True

    for c in range(0,num_colors):
        
        img = cv2.imread(img_name,cv2.IMREAD_COLOR)
        height = img.shape[0]
        width = img.shape[1]


        # TODO: Insert your algorithm here

        lower = color_ranges[(c*2)+1]
        upper = color_ranges[(c*2)]

        # convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # create a mask
        if (color_names[c] == 'red'):
            mask_bottom_red = cv2.inRange(hsv, bottom_lower_red, bottom_upper_red)
            mask_top_red = cv2.inRange(hsv, top_lower_red, top_upper_red)
            
            mask = cv2.add(mask_bottom_red, mask_top_red)

            kernel_size = 4
            kernel_shape = morph_shape(2)
            mask = opening(mask,kernel_size, kernel_shape)
            mask = closing(mask,kernel_size, kernel_shape)
            mask = erosion(mask,kernel_size, kernel_shape, 2)
            mask = dilation(mask,kernel_size, kernel_shape, 3)
        
        elif (color_names[c] == 'yellow'):
            mask = cv2.inRange(hsv, lower, upper)
            kernel_size = 2
            kernel_shape = morph_shape(2)
            mask = opening(mask,kernel_size, kernel_shape)
            mask = closing(mask,kernel_size, kernel_shape)
            mask = erosion(mask,kernel_size, kernel_shape, 6)
            mask = dilation(mask,kernel_size, kernel_shape, 3)

        else: 
            mask = cv2.inRange(hsv, lower, upper)

            kernel_size = 3
            kernel_shape = morph_shape(2)
            mask = opening(mask,kernel_size, kernel_shape)
            mask = closing(mask,kernel_size, kernel_shape)

            if (color_names[c] == 'green'):
                mask = erosion(mask,kernel_size, kernel_shape, 3)
                mask = dilation(mask,kernel_size, kernel_shape, 2)
            elif (color_names[c] == 'blue'):
                mask = erosion(mask,kernel_size, kernel_shape, 4)
                mask = dilation(mask,kernel_size, kernel_shape, 3)
            elif (color_names[c] == 'white'):
                mask = erosion(mask,kernel_size, kernel_shape, 4)
                mask = dilation(mask,kernel_size, kernel_shape, 2)
            elif (color_names[c] == 'pink'):
                mask = erosion(mask,kernel_size, kernel_shape, 5)
                mask = dilation(mask,kernel_size, kernel_shape, 4)

        cv2.imshow("Maske",mask)

        cv2.waitKey(0)

        # find connected components
        connectivity = 4
        roundness = 1
        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(mask,connectivity,cv2.CV_32S)
        num_rejected = 1
        

        for i in range(1,num_labels):
            # check size and roundness as plausibility
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w < min_size or h < min_size:
                num_rejected += 1
                continue # found component is too small to be correct 
            if w > h:
                roundness = 1.0 / (w/h)
            elif h > w:
                roundness = 1.0 / (h/w)  
            if (roundness < .8):
                num_rejected += 1
                continue # ratio of width and height is not suitable

            center = centroids[i]
            center = np.round(center)
            center = center.astype(int)

            cv2.circle(img,center,circle_size,red_BGR,circle_thickness)

            cv2.rectangle(img, (x, y), (x + w, y + h), green_BGR, 3) 



        num_final_labels = num_labels-num_rejected

        success = (num_final_labels == int(gt_list[c]))
        
        if success:
            print('We have found all', str(num_final_labels),'/',str(gt_list[c]), color_names[c],'chewing gum balls. Yeah!')
            foo = 0
        elif (num_final_labels > int(gt_list[c])):
            print('We have found too many (', str(num_final_labels),'/',str(gt_list[c]),') candidates for', color_names[c],'chewing gum balls. Damn!')
            all_colors_correct = False
        else:
            print('We have not found enough (', str(num_final_labels),'/',str(gt_list[c]),') candidates for', color_names[c],'chewing gum balls. Damn!')
            all_colors_correct = False
        
        # debug output of the test images
        if ((img_name == 'images\chewing_gum_balls04.jpg') 
            or (img_name == 'images\chewing_gum_balls08.jpg') 
            or (img_name == 'images\chewing_gum_balls12.jpg')):
            # show the original image with drawings in one window
            cv2.imshow('Original image', img)
            # show other images?

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    if all_colors_correct:
        num_test_images_succeeded += 1
        print ('Yeah, all colored objects have been found correctly in ',img_name)

print ('Test result:', str(num_test_images_succeeded),'test images succeeded.')