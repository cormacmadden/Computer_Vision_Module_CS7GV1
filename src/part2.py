from ctypes import resize
from MidTermAssignment import *
from os import listdir
from os.path import isfile, join

def run():
    cv.destroyAllWindows()

    mypath='../Photos/Part2'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(join(mypath,onlyfiles[n]),cv.IMREAD_UNCHANGED )
    
    images[0] = scale_image(images[0],0.15)
    images[1] = scale_image(images[1],0.3)
    images[2] = scale_image(images[2],0.1)
    
    noisyImage = images[0]
    noisyImage = scale_image(noisyImage,0.2)
    ksize = (10, 10)
    
    for img in images:
        img = cv.resize(img, (540, 960))
        blurred = kernel_transform(img,ksize)
        Hori = np.concatenate((Hori, blurred), axis=1)
        cv.imshow("Before/After", Hori)
        
    blurry_image = cv.blur(noisyImage, ksize)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()