from ctypes import resize
from MidTermAssignment import *
import os

def run():
    
    cv2.destroyAllWindows()
    fileDir = os.path.dirname(__file__)
    mypath = os.path.join(fileDir, '../Photos/Part2')
    onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(os.path.join(mypath,onlyfiles[n]),cv2.IMREAD_UNCHANGED )
        
    #images[0] = subsample_image(images[0])
    #images[1] = subsample_image(images[1])
    #images[2] = subsample_image(images[2])

    kernalSize = 1
    #bigger filter will make filter more obvious
    ##blurring
    '''
    image = images[1]
    boxBlurred = box_blur(image,kernalSize)
    cv2.putText(boxBlurred,"Box Blur, Kernal Size = 9",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    gaussianBlurred = gaussian_blur(image,kernalSize)
    cv2.putText(gaussianBlurred,"Gaussian Blur, Kernal Size = 9",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    Hori = np.concatenate((image, boxBlurred), axis=1)
    Hori = np.concatenate((Hori, gaussianBlurred), axis=1)
    '''
    
    ##sharpening
    image = images[0]
    cv2.putText(image,"Original Image",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    sharpened = bilateral_filter(image,13)
    cv2.putText(sharpened,"gaussian_blur, Kernal Size = 13",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    
    #sobeled = sobel_filter(image,kernalSize)
    #sobeled = greyscale_to_rgb(sobeled)
    #cv2.putText(sobeled,"Sobel Filter, Kernal Size = 9",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    Hori = np.concatenate((image, sharpened), axis=1)
    #Hori = np.concatenate((Hori, sobeled), axis=1)
    cv2.imshow("Sharpening", Hori)
    '''
    ##non-Linear Diffusion
    image = images[3]
    boxBlurred = median_filter(image,kernalSize)
    cv2.putText(boxBlurred,"Median Filter, Kernal Size = 9",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    gaussianBlurred = bilateral_filter(image,kernalSize)
    cv2.putText(gaussianBlurred,"Bilateral Filter, Kernal Size = 9",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    Hori = np.concatenate((image, boxBlurred), axis=1)
    Hori = np.concatenate((Hori, gaussianBlurred), axis=1)
    cv2.imshow("Blurring", Hori)
    '''
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()