from ctypes import resize
from MidTermAssignment import *
from os import listdir
from os.path import isfile, join

def run():
    cv.destroyAllWindows()
    dataset = load_sample_images()
    images = dataset['images']
    
    mypath='../Photos'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(join(mypath,onlyfiles[n]),cv.IMREAD_UNCHANGED )
        
    noisyImage = images[0]
    noisyImage = scale_image(noisyImage,0.2)
    ksize = (10, 10)
    #output = saturateImage(img,2)
    
    blurry_image = cv.blur(noisyImage, ksize)
    
    #Hori = np.concatenate((noisyImage, blurry_image), axis=1)
    cv.imshow('Images' , noisyImage)
    cv.imshow("Processed", blurry_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()
    
