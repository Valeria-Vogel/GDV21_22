''' This code is based on the stackoverflow answer from Fred Weinhaus: https://stackoverflow.com/a/59995542  '''
import cv2
import numpy as np

# global helper variables
window_width = 640
window_height = 480

def getFrequencies(image):   
    # convert image to floats and do dft saving as complex output
    dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    
    # apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft)
        
    # extract magnitude and phase images
    mag, phase = cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1])

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mag) 
    print('mag:', min_val, max_val, min_loc, max_loc)
    print(mag.shape)
 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(phase) 
    print('phase:', min_val, max_val, min_loc, max_loc)
    print(phase.shape)

    mag, phase = changeMagAndPhase(mag, phase)

    # get spectrum for viewing only
    spec = ((1/20) * np.log(mag))    
    
    # Return the resulting image (as well as the magnitude and phase for the inverse)
    return spec, mag, phase

def createFromSpectrum(mag, phase):
    # convert magnitude and phase into cartesian real and imaginary components
    real, imag = cv2.polarToCart(mag, phase)

    # combine cartesian components into one complex image
    back = cv2.merge([real, imag])
    
    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(back)

    # do idft saving as complex output
    img_back = cv2.idft(back_ishift)
    # combine complex components into original image again
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    # re-normalize to 8-bits
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_back
def changeMagAndPhase(mag, phase): 
    # mag[:,100:115]= 0
    # mag[140:150,:]= 0
    # mag[300:315,:]= 0
    # mag[:,360:380]= 0
    mag[220:265,300:345]= 0
    return mag, phase

def main():
    """ Load an image, compute frequency domain image from it and display both or vice versa """
    image_name = 'images\Beispielbilder-der-Nikon-D850-1024x683-e2452dee30cda9f1.jpg'

    # Load the image.
    image = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (window_width,window_height))

    # show the original image
    title_original = 'Original image'
    cv2.namedWindow(title_original, cv2.WINDOW_NORMAL) # Note that window parameters have no effect on MacOS
    cv2.resizeWindow(title_original,window_width,window_height)
    cv2.imshow(title_original,image)
        
    result, mag, phase = getFrequencies(image)
    #result = np.zeros((window_height,window_width),np.uint8)

    
    # show the resulting image
    title_result = 'Frequencies image'
    cv2.namedWindow(title_result, cv2.WINDOW_NORMAL) # Note that window parameters have no effect on MacOS
    cv2.resizeWindow(title_result,window_width,window_height)
    cv2.imshow(title_result,result)

    back = createFromSpectrum(mag, phase)
    #back = np.zeros((window_height,window_width),np.uint8)

    # and compute image back from frequencies
    title_back = 'Reconstructed image'
    cv2.namedWindow(title_back, cv2.WINDOW_NORMAL) # Note that window parameters have no effect on MacOS
    cv2.resizeWindow(title_back,window_width,window_height)
    cv2.imshow(title_back,back)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

if (__name__ == '__main__'):
    main()


