from ctypes import resize
from MidTermAssignment import *
from os import listdir
from os.path import isfile, join
import time

def run():
    cv.destroyAllWindows()
    
    mypath='../Photos/Part3'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv.imread(join(mypath,onlyfiles[n]),cv.IMREAD_UNCHANGED )
    
    
    
    image = images[1]
    cv.imshow("Before", image)

    #for i in range(1,10):
    #minify_image(image,i/100)
    
    scaled = scale_linear_interpolation(image,0.25)
    
    cv.imshow("After", scaled)
    time.sleep(1.0)
    cv.waitKey(0)
    cv.destroyAllWindowss()

    #cv.waitKey(0)
    #cv.destroyAllWindows()



if __name__ == '__main__':
    run()