import cv2
from pseudoRGB import pseudoRGB
from resizeToFit import *
import keras.backend as K
def preprocessing (x, y = None, resizeTo = (224,224)):
    x = resizeToFit (x, resizeTo)
    x = pseudoRGB (x, "clahe")/255
    if y is not None:
        y = resizeToFit (y, resizeTo)
        y = y.astype('float32')/255
        return x, y
    return x
   

def postprocessing (x, imageShape, y = None, method = "crop", visualize = False):
    if method == "resize":
        x = resizeToFit (x, imageShape)
        if y is not None:
            y = resizeToFit (y, imageShape)
            y = y.astype('float32')/255
            return x, y
        return x
                
    if method == "crop":
        offsetRow = (x.shape[0] - imageShape[0])//2
        offsetCol = (x.shape[1] - imageShape[1])//2
        x = x[offsetRow:offsetRow+imageShape[0], offsetCol:offsetCol + imageShape[1], :].copy()

        if visualize == True:
            fig2 = plt.figure(figsize = (10,5)) # create a 5 x 5 figure 
            for i in range(0, 9):
                pyplot.subplot(1, 2,  1)
                pyplot.imshow(x)
            pyplot.show()

        if y is not None:
            y = resizeToFit (y, imageShape)
            y = y.astype('float32')/255
            return x, y
        return x     

def clahe_augment(img):
    clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    clahe_medium = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_high = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    img_low = clahe_low.apply(img)
    img_medium = clahe_medium.apply(img)
    img_high = clahe_high.apply(img)
    augmented_img = np.array([img_low, img_medium, img_high])
    augmented_img = np.swapaxes(augmented_img,0,1)
    augmented_img = np.swapaxes(augmented_img,1,2)
    return augmented_img

def contrast_augment(img):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    augmented_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return augmented_img















