from macpath import split
from unittest import result
from sklearn.datasets import load_sample_images
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
import math

def brighten(img, amt):
    output = img
    r,g,b = split(img)
    
    b1 = (b[:,:]/255) * amt
    g1 = (g[:,:]/255) * amt
    r1 = (r[:,:]/255) * amt
    
    b = b[:,:] + b1
    r = r[:,:] + r1
    g = g[:,:] + g1

    b = np.clip(b,0,255)
    r = np.clip(r,0,255)
    g = np.clip(g,0,255)
    
    b=b.astype('uint8')
    g=g.astype('uint8')
    r=r.astype('uint8')   
     
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
        
    return output

def sat_image(img,amt):
    #h,s,v = cv.split(img)
    processedImage=img[:,:].astype('float32')
    processedImage = processedImage[:,:]+amt
    #s=s*amt
    processedImage = np.clip(processedImage,0,255)
    processedImage=processedImage.astype('uint8')
    #img = cv.merge([h,s,v])
    #processedImage = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return processedImage

def greyscale(img):
    r,g,b = split(img)
    b=b.astype('float32')
    g=g.astype('float32')
    r=r.astype('float32')

    b = b/255
    g = g/255
    r = r/255
    
    #Gamma transformation 
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if(r[i,j] < 0.04045): r[i,j] = r[i,j]/12.92
            if(b[i,j] < 0.04045): b[i,j] = b[i,j]/12.92
            if(g[i,j] < 0.04045): g[i,j] = g[i,j]/12.92
            
    r = ((r+0.055)/1.055)**2.4
    g = ((g+0.055)/1.055)**2.4
    b = ((b+0.055)/1.055)**2.4
    
    Y = 0.2126*r + 0.7152*g + 0.0722*b
    output = np.zeros((img.shape[0],img.shape[1],1))
    ytest = 1.055*(Y[0,0]**(1/2.4))-0.055
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if Y[i,j] <= 0.0031308:
                output[i,j] = 12.92*Y[i,j]
            else:
                output[i,j] = (1.055*(Y[i,j]**(1/2.4)))-0.055
    
    output = output*255        
    output = np.clip(output,0,255)
    output = output.astype('uint8')
    return output

def greyscale_to_rgb(img):
    output = np.zeros((img.shape[0],img.shape[1],3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j,0] = img[i,j]
            output[i,j,1] = img[i,j]
            output[i,j,2] = img[i,j]
            
    output = np.clip(output,0,255)
    output=output.astype('uint8')
    return output

def contrast_image(img,contrast, brightness):
    #new_image = np.zeros(img.shape, img.dtype)
    output = img
    r,g,b = split(output)
    b=b.astype('float32')
    g=g.astype('float32')
    r=r.astype('float32')
    
    r = (r-128) * contrast + 128
    g = (g-128) * contrast + 128
    b = (b-128) * contrast + 128
    
    b = np.clip(b,0,255)
    r = np.clip(r,0,255)
    g = np.clip(g,0,255)

    b=b.astype('uint8')
    g=g.astype('uint8')
    r=r.astype('uint8')
     
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    
    output = np.clip(output,0,255)
    output=output.astype('uint8')
    return output

def threshold(img, amt):
    greyed = greyscale(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(greyed[i,j]<amt):greyed[i,j] = 0
            else: greyed[i,j] = 1
    
    greyed = greyed*255
    return greyscale_to_rgb(greyed)
    
def split(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    rgb = r,g,b
    return rgb

def solarization(img, amt = 1):
    output = brighten(img, 100)
    r,g,b = split(output)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if ((r[i,j] == 255) and (g[i,j] == 255) and (b[i,j] == 255 )):
                output[i][j] = [0,0,0]
                
    output = output.astype('uint8')
    return output

def scale_image(img, percentage: float):
    width = int(img.shape[1] * percentage)
    height = int(img.shape[0] * percentage)
    dim = (width, height)
    return cv.resize(img,dim,interpolation = cv.INTER_NEAREST)

def image_temp(img, amt):
    
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    b=b.astype('float32')
    r=r.astype('float32')
    
    b = b[:,:] - amt
    r = r[:,:] + amt
    
    b = np.clip(b,0,255)
    r = np.clip(r,0,255)
    
    b=b.astype('uint8')
    r=r.astype('uint8')

    output = np.zeros((img.shape[0],img.shape[1],3))
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    
    return output

def image_tint(img, amt):
    
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    g=g.astype('float32')
    
    g = g[:,:] + amt
    
    g = np.clip(g,0,255)
    
    g=g.astype('uint8')

    output = np.zeros((img.shape[0],img.shape[1],3))
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    
    return output

def box_blur(img, amt):
    arr = np.zeros((amt,amt,3))
    arr = 1
    arr = arr/(amt**2)
    output = img * 0
    pad = (amt - 1) // 2
	#output = np.zeros((iH, iW), dtype="float32")
    
    for i in range(0, img.shape[0],1):
        for j in range(0, img.shape[1],1):
            #duplicate of base image, size of kernel
            test = i-pad
            test2 = i+pad
            test3 = j-pad
            test4 = j+pad
            
            test = np.clip(test,0,img.shape[0])
            test2 = np.clip(test2,0,img.shape[0])
            test3 = np.clip(test3,0,img.shape[1])
            test4 = np.clip(test4,0,img.shape[1])
                       
            roi = np.zeros((amt,amt,3))
            roi = img[(test):(test2),(test3):(test4)]
            
            k1 = (roi * arr)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            output[i][j]=k3
    return output

def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution3D(image, kernel)

def sobel_filter(image,amt):
    image = greyscale(image)
    
    kernel_hori = np.zeros((3,3,1))
    kernel_hori[:,0] = -1
    kernel_hori[:,2] = 1
    kernel_hori *= amt
    img_hori = convolution_grey(image,kernel_hori)
    
    kernel_vert = np.zeros((3,3,1))
    kernel_vert[0,:] = -1
    kernel_vert[2,:] = 1
    kernel_vert *= amt
    img_vert = convolution_grey(image,kernel_vert)
    
    img_vert=img_vert.astype('uint8')
    img_hori=img_hori.astype('uint8')
    
    output = (img_vert/2) + (img_hori/2)
    output = np.clip(output,0,255)
    output=output.astype('uint8')
    return output

def convolution3D(image, kernel):
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    show_kernel(kernel[:,:,0],"kernel")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            #show_region(roi[:,:,0],"ROI")
            k1 = (kernel * roi)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            k3 = np.clip(k3,0,255)
            output[i][j]=k3
    output=output.astype('uint8')
    return output

def convolution2D(image, kernel):
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    show_kernel(kernel,"kernel")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            show_region(roi[:,:,0],"ROI")
            k1 = (kernel * roi)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            k3 = np.clip(k3,0,255)
            output[i][j]=k3
    output=output.astype('uint8')
    #cv.imshow("Test", output)
    return output

def convolution_grey(image, kernel):
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    show_kernel(kernel,"kernel")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1],:]
            show_region(roi[:,:],"ROI")
            k1 = kernel * roi
            k4 = abs(np.sum(k1))
            k4 = np.clip(k4,0,255)
            if k4 > 50:
                print(k4 , " \n")
            output[i,j] = k4 
        #cv.imshow("Test", output)
    return output

def sharpening_filter(image, amt):
    kernel = np.zeros((3,3,1))
    kernel[1,0,0]=-1
    kernel[0,1,0]=-1
    kernel[1,2,0]=-1
    kernel[2,1,0]=-1
    kernel[1,1,0]=5
    kernel*=amt
    output = convolution2D(image,kernel)
    return output

def bilateral_filter(image, kernel_size):
    
    kernel = gaussian_kernel(kernel_size)
    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]
    
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    

    show_kernel(kernel[:,:,0],"kernel")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel_width, j:j + kernel_height,:]
            centerpix = roi[kernel_width//2,kernel_height//2,:]
            centerpixRGB = np.sum(centerpix, axis = 0)
            kernel = gaussian_kernel(kernel_size)
            for k in range(kernel_width):
                for l in range(kernel_height):
                    pixRGBsum = np.sum(roi[k,l], axis = 0)
                    diff = pixRGBsum - centerpixRGB
                    if abs(diff) > 10 :
                        kernel [k,l] = 0
            
            show_kernel(kernel[:,:,0], "Kernel")
            #     ####distance=(R1+G1+B1)-(R2+G2+B2)
            #kernel[:,:,rgb] = normalize(kernel[:,:,rgb],0,1)
            k1 = (kernel * roi)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            k3 = np.clip(k3,0,255)
            output[i][j][:]=k3
            show_region(roi, "ROI")
        show_progress(output,"TempOutput")
        
    output = np.clip(output,0,255)
    output=output.astype('uint8')
    return output

def median_filter(image, kernel_size):
    kernel = np.zeros((kernel_size,kernel_size,3))
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            roi = roi**2
            for k in range(0,3):
                pixel = np.median(roi[:,:,k])
                pixel = np.sqrt(pixel)
                pixel = np.clip(pixel,0,255)
                output[i,j,k]=pixel

    output=output.astype('uint8')
    return output

def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    one = np.sum(kernel_1D)

    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    sum = np.sum(kernel_2D)

            
    for i in range(kernel_2D.shape[0]):
        for j in range(kernel_2D.shape[1]): 
            kernel_2D[i,j] /= sum
            
    #kernel_2D *= 1.0 / kernel_2D.max()
    
    kernel_3D = np.zeros((size,size,3))
    kernel_3D[:,:,0] = kernel_2D[:,:]
    kernel_3D[:,:,1] = kernel_2D[:,:]
    kernel_3D[:,:,2] = kernel_2D[:,:]
 
    return kernel_3D

def show_kernel(kernel, title):
    kernel = kernel*255*40
    kernel = np.clip(kernel,0,255)
    img = np.zeros((kernel.shape[0],kernel.shape[1]),dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]= kernel[i,j]
        
    img=img.astype('uint8')
    img = scale_image(img,10)
    cv.imshow(title, img)

def show_region(region, title):
    region = scale_image(region,15)
    region=region.astype('uint8')
    cv.imshow(title, region)
    
def show_progress(region, title):
    region = scale_image(region,10)
    region=region.astype('uint8')
    cv.imshow(title, region)
    
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    max1 = arr.max()
    min1 = arr.min()
    diff_arr = max1 - min1
    for i in arr:
        temp = (((i - arr.min())*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def pad_image(image,kernel):
    
    image_width = image.shape[0]
    image_height = image.shape[1]
    
    padx = (kernel.shape[0] - 1) // 2
    pady = (kernel.shape[1] - 1) // 2
    

    #blank image that's padded on x and y
    padded_image = np.zeros((image_width + (2 * pady), image_height + (2 * padx),image.shape[2]))
        
    #filled image that's padded on x and y
    padded_image[padx:padded_image.shape[0] - padx, pady:padded_image.shape[1] - pady] = image
    
    for i in range(padx):
        #left padding
        padded_image[i,pady:padded_image.shape[1]-pady] = image[0,:]
        
        #right padding
        padded_image[(i+image_width+padx),pady:padded_image.shape[1]-pady] = image[image_width-4,:]
        
    for j in range(pady):
        #top padding
        padded_image[padx:padded_image.shape[0]-padx,j] = image[:,0]
        #bottom padding
        padded_image[padx:padded_image.shape[0]-padx,j+image_height+pady] = image[:,image_height-4]
        
    return padded_image

def subsample_image(image):
    kernel = np.zeros((2,2,3))
    output = np.zeros((int((image.shape[0]/2) +1),int((image.shape[1]/2)+1),3))
    padded_image = pad_image(image, kernel)
    for i in range(0,image.shape[0],2):
        for j in range(0,image.shape[1],2):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            roi = roi**2
            for k in range(0,3):
                pixel = np.median(roi[:,:,k])
                pixel = np.sqrt(pixel)
                pixel = np.clip(pixel,0,255)
                output[int(i/2),int(j/2),k]=pixel

    output=output.astype('uint8')
    return output

def gaussian_pyramid(image):
    image = gaussian_blur(image,3)
    image = subsample_image(image)
    return image

def scale_NN(image, amt):
    output = np.zeros((int(image.shape[0]*amt),int(image.shape[1]*amt),3))
    for i in range(0,output.shape[0]):
        for j in range(0,output.shape[1]):
            transformedI = int((i/output.shape[0])*image.shape[0])
            transformedJ = int((j/output.shape[1])*image.shape[1])
            output[i,j] = image[transformedI,transformedJ]
    output=output.astype('uint8')
    return output

def scale_bilinear_interpolation(image, amt):
    output = np.zeros((int(image.shape[0]*amt),int(image.shape[1]*amt),3))
    for i in range(0,output.shape[0]):
        for j in range(0,output.shape[1]):
            transformedI = (i/output.shape[0])*image.shape[0]
            transformedJ = (j/output.shape[1])*image.shape[1]
            for rgb in range(0,3):
                if int(math.ceil(transformedI)) >= image.shape[0] :
                    iceil = image.shape[0]-1
                else:
                    iceil = int(math.ceil(transformedI))
                    
                if int(math.ceil(transformedJ)) >= image.shape[1] :
                    jceil = image.shape[1]-1
                else:
                    jceil = int(math.ceil(transformedJ))
                    
                ifloor = int(math.floor(transformedI))
                jfloor = int(math.floor(transformedJ))
                x1y1 = image[ifloor,jfloor,rgb]
                x2y2 = image[iceil,jceil,rgb]
                x1y2 = image[ifloor,jceil,rgb]
                x2y1 = image[iceil,jfloor,rgb]
                iweightedAvg1 = x1y1*1-(transformedI%1) + x2y1*transformedI%1
                iweightedAvg2 = x1y2*1-(transformedI%1) + x2y2*transformedI%1
                finalPix = iweightedAvg1*(1-(transformedJ%1)) + iweightedAvg2*(transformedJ%1)
                
                output[i,j,rgb] = finalPix
    output=output.astype('uint8')
    return output

def bilateral_blur_filter_individual_channels(image, kernel_size):
    
    kernel = gaussian_kernel(kernel_size)
    
    image_width = image.shape[0]
    image_height = image.shape[1]
    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]
 
    output = np.zeros(image.shape)
 
    padx = (kernel.shape[0] - 1) // 2
    pady = (kernel.shape[1] - 1) // 2
 
    #blank image that's padded on x and y
    padded_image = np.zeros((image_width + (2 * pady), image_height + (2 * padx),3))
    #filled image that's padded on x and y
    padded_image[padx:padded_image.shape[0] - padx, pady:padded_image.shape[1] - pady] = image
    
    for i in range(padx):
        #left padding
        padded_image[i,pady:padded_image.shape[1]-pady] = image[0,:]
        #right padding
        padded_image[(i+image_width+padx),pady:padded_image.shape[1]-pady] = image[image_width-4,:]
        
    for j in range(pady):
        #top padding
        padded_image[padx:padded_image.shape[0]-padx,j] = image[:,0]
        #bottom padding
        padded_image[padx:padded_image.shape[0]-padx,j+image_height+pady] = image[:,image_height-4]

    show_kernel(kernel[:,:,0],"kernel")
    for i in range(image_width):
        for j in range(image_height):
            roi = padded_image[i:i + kernel_width, j:j + kernel_height,:]
            centerpix = roi[kernel_width//2,kernel_height//2,:]
            ##Code Here
            for rgb in range(0,3):
                kernel = gaussian_kernel(kernel_size)
                for k in range(kernel_width):
                    for l in range(kernel_height):
                        diff = roi[k,l,rgb] - centerpix[rgb]
                        if abs(diff) > 10 :
                            kernel [k,l,rgb] = 0
                
                show_kernel(kernel[:,:,rgb], "Kernel")
                #kernel[:,:,rgb] = normalize(kernel[:,:,rgb],0,1)
                k1 = (kernel[:,:,rgb] * roi[:,:,rgb])
                k2 = np.sum(k1, axis = 0)
                k3 = np.sum(k2, axis = 0)
                k3 = np.clip(k3,0,255)
                output[i,j,rgb]=k3
            show_region(roi, "ROI")
        #show_progress(output,"TempOutput")
    return output

def image_flip(img):
    
    #img[i,j] = 
    return img

def invert_colors(img):
    return img