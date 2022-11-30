from ctypes import resize
from MidTermAssignment import *
import os
import time

def run():

    cv2.destroyAllWindows()
    fileDir = os.path.dirname(__file__)
    mypath = os.path.join(fileDir, '../Photos/Part3')
    onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(os.path.join(mypath,onlyfiles[n]),cv2.IMREAD_UNCHANGED )
    
    #subsampling
    #for image in images:
    image = images[3]
    
    image = gaussian_pyramid_bilateral(image)
    
    #cv2.imshow("Before", image)
    #minimized1 = gaussian_pyramid_bilateral(image)
    #minimized2 = gaussian_pyramid_bilateral(minimized1)
    #minimized2 = scale_NN(minimized2, 4)
    
    #minimized1 = scale_NN(minimized1, 2)
    #minimized2 = scale_NN(minimized2, 2)
    
    #cv2.imshow("Pyramid1", minimized1)
    #cv2.imshow("Pyramid2", minimized2)
    #cv2.imshow("Pyramid3", minimized3)
        

    #sub1 = subsample_image(image)
    #sub2 = subsample_image(sub1)
    #sub3 = subsample_image(sub2)
    image = scale_NN(image,2)
    #sub2 = scale_NN(minimized2,4)
    #sub3 = scale_NN(minimized3,8)
    
    cv2.putText(image,"Gaussian Pyramid Subsampled, halfed",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    cv2.imshow("ssPyramid1", image)   
    #cv2.putText(sub1,"Bilinear Interpolation, 1st Downsample",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    #cv2.imshow("ssPyramid1", sub1)
    #cv2.putText(sub2,"Bilinear Interpolation, 2nd Downsample",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    #cv2.imshow("ssPyramid2", sub2)
    #cv2.putText(sub2,"Bilinear Interpolation, 3rd Downsample",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    #cv2.imshow("Pyramid3", sub3)
    '''
    #NNN
    
    sub1 = scale_bilinear_interpolation(image,2)
    sub2 = scale_bilinear_interpolation(image,4)
    sub3 = scale_bilinear_interpolation(image,8)
    
    cv2.putText(image,"Bilinear Interpolation, Original Image",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    cv2.imshow("ssPyramid", image)   
    cv2.putText(sub1,"Bilinear Interpolation, 2x scale",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    cv2.imshow("ssPyramid1", sub1)
    cv2.putText(sub2,"Bilinear Interpolation, 4x scale",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    cv2.imshow("ssPyramid2", sub2)
    cv2.putText(sub3,"Bilinear Interpolation, 8x scale",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    cv2.imshow("Pyramid3", sub3)
    
    cv2.waitKey(0)
    cv2.destroyAllWindowss()
    #for i in range(1,10):
    #minify_image(image,i/100)
    '''


    cv2.waitKey(0)
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    run()