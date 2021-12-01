# Template matching, originally with objects from the image. Typical example is counting blood cells
import cv2

use_color = True

if use_color:
    # load image and template image, note that the template has been manually cut out of the image
    img = cv2.imread('images\chewing_gum_balls06.jpg', cv2.IMREAD_COLOR)
    tamplate = cv2.imread('images\cgb_pink.jpg', cv2.IMREAD_COLOR)

    # read shape of the template and original image
    H, W, C = img.shape
    h, w, c = tamplate.shape
    
else:
    # load image and template image, note that the template has been manually cut out of the image
    img = cv2.imread('images\chewing_gum_balls06.jpg', cv2.IMREAD_GRAYSCALE)
    tamplate = cv2.imread('images\cgb_pink.jpg', cv2.IMREAD_GRAYSCALE)
    
    # read shape of the template and original image
    H, W = img.shape
    h, w = tamplate.shape

# Define template matching methods, see https://docs.opencv.org/4.5.3/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d for the math behind each method
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
method_names = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# loop over all methods in order to compare them
for method,method_name in zip(methods,method_names):

    # work on a new image each time
    img_2 = img.copy()

    # do the template matching
    result = cv2.matchTemplate(img_2, tamplate, method)
    result2 = result.copy()
    result2 = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # get the best match location
    min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)
    print(min_value, max_value, min_location, max_location)

    # draw rectangle at found location

    #cv2.rectangle(img_2 ,min_location , max_location , (255, 255, 255), 3)
    cv2.rectangle(img_2 ,min_location , (min_location[0]*w ,min_location[1]*h) , (50, 250, 50), 2)
    cv2.rectangle(img_2 ,max_location , (max_location[0]*w ,max_location[1]*h) , (50, 150, 250), 2)
    cv2.rectangle(result2 ,min_location , (min_location[0]*w ,min_location[1]*h) , (50, 250, 50), 2)
    cv2.rectangle(result2 ,max_location , (max_location[0]*w ,max_location[1]*h) , (50, 150, 250), 2)
    
    
    # show original image with found location
    cv2.imshow('Original', img_2)
    
    # show image with the template matching result for all pixels
    result_title = 'Result with %s' % method_name
    cv2.imshow(result_title, result)
    cv2.waitKey(0)
    cv2.destroyWindow(result_title)

cv2.destroyAllWindows()