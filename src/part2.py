from ctypes import resize
from MidTermAssignment import *
import os

def run():
    
    cv.destroyAllWindows()
    fileDir = os.path.dirname(__file__)
    mypath = os.path.join(fileDir, '../Photos/Part2')
    onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(os.path.join(mypath,onlyfiles[n]),cv.IMREAD_UNCHANGED )
        
    #images[0] = scale_image(images[0],0.15)
    #images[1] = scale_image(images[1],0.3)
    #images[2] = scale_image(images[2],0.1)
    #images[3] = scale_image(images[3],0.25)

    noisyImage = images[0]
    kernalSize = 9
    #images still like 2x too big
    #bigger filter will make filter more obvious
    ##blurring
    image = images[1]
    boxBlurred = box_blur(image,7)
    gaussianBlurred = gaussian_blur(image,7)
    Hori = np.concatenate((image, boxBlurred), axis=1)
    Hori = np.concatenate((Hori, gaussianBlurred), axis=1)
    cv.imshow("Blurring", Hori)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    run()