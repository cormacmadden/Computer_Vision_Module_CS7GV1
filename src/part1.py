from ctypes import resize
from MidTermAssignment import *
import os

def run():
    cv.destroyAllWindows()
    fileDir = os.path.dirname(__file__)
    mypath = os.path.join(fileDir, '../Photos/Part1')
    onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(os.path.join(mypath,onlyfiles[n]),cv.IMREAD_UNCHANGED )
        
    images[0] = scale_image(images[0],0.15)
    images[1] = scale_image(images[1],0.3)
    images[2] = scale_image(images[2],0.1)
    
    for img in images:
        brightened = brighten(img,100)
        warmed = image_temp(img, 50)
        cooled = image_temp(img, -50)
        tinted = image_tint(img, -50)
        contrasted = contrast_image(img,2,5)
        solarized = solarization(img)
        greyed = greyscale(img)
        greyed = greyscale_to_rgb(greyed)
        threshed = threshold(img, 128)
        flipped = image_flip(img)
        Hori = np.concatenate((img, brightened), axis=1)
        Hori = np.concatenate((Hori, cooled), axis=1)
        Hori = np.concatenate((Hori, warmed), axis=1)
        Hori = np.concatenate((Hori, tinted), axis=1)
        Hori = np.concatenate((Hori, contrasted), axis=1)
        Hori = np.concatenate((Hori, solarized), axis=1)
        Hori = np.concatenate((Hori, greyed), axis=1)
        Hori = np.concatenate((Hori, threshed), axis=1)
        cv.imshow("Before/After", Hori)

        #cv.imshow("Before/After", greyed) 
    
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()