from ctypes import resize
from MidTermAssignment import *
import os
import time

def run():

    cv.destroyAllWindows()
    fileDir = os.path.dirname(__file__)
    mypath = os.path.join(fileDir, '../Photos/Part3')
    onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(os.path.join(mypath,onlyfiles[n]),cv.IMREAD_UNCHANGED )
    
    
    ##blurring
    for image in images:
        cv.imshow("Before", image)
        boxBlurred = box_blur(image,5)
        gaussianBlurred = gaussian_blur(image,5)
        Hori = np.concatenate((image, boxBlurred), axis=1)
        Hori = np.concatenate((Hori, gaussianBlurred), axis=1)
        cv.imshow("Blurring", Hori)
        cv.waitKey(0)
        cv.destroyAllWindowss()
    #for i in range(1,10):
    #minify_image(image,i/100)
    


    #cv.waitKey(0)
    #cv.destroyAllWindows()



if __name__ == '__main__':
    run()