from ctypes import resize
from MidTermAssignment import *
from os import listdir
from os.path import isfile, join

def run():
    cv.destroyAllWindows()

    mypath='../Photos/Part1'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(join(mypath,onlyfiles[n]),cv.IMREAD_UNCHANGED )
        
    images[0] = scale_image(images[0],0.15)
    images[1] = scale_image(images[1],0.3)
    images[2] = scale_image(images[2],0.1)
    
    for img in images:
        brightened = brighten_image(img,50)
        warmed = image_temp(img, 50)
        cooled = image_temp(img, -50)
        tinted = image_tint(img, -50)
        saturated = sat_image(img,2)
        cv.imshow("cooled", cooled) 
        contrasted = contrast_image(img,1.5,50)
        cv.imshow("contrasted", contrasted) 
        solarized = solarization(img)
        Hori = np.concatenate((img, brightened), axis=1)
        Hori = np.concatenate((Hori, cooled), axis=1)
        Hori = np.concatenate((Hori, warmed), axis=1)
        Hori = np.concatenate((Hori, tinted), axis=1)
        cv.imshow("Before/After", Hori)     
    
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()