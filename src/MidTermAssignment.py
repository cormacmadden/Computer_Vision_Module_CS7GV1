import math
from copy import deepcopy

import cv2
import numpy as np

position = (10,20)
def clip(r,g,b):
    b = np.clip(b,0,255)
    r = np.clip(r,0,255)
    g = np.clip(g,0,255)
    return r,g,b

def to_uint8(r,g,b):
    b=b.astype('uint8')
    g=g.astype('uint8')
    r=r.astype('uint8')  
    return r,g,b
    
def split(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    rgb = r,g,b
    return rgb

def sat_image(img,amt):
    #h,s,v = cv2.split(img)
    processedImage=img[:,:].astype('float32')
    processedImage = processedImage[:,:]+amt
    #s=s*amt
    processedImage = np.clip(processedImage,0,255)
    processedImage=processedImage.astype('uint8')
    #img = cv2.merge([h,s,v])
    #processedImage = cv2.cv2tColor(img, cv2.COLOR_HSV2BGR)
    return processedImage

def brighten(img, amt):
    output = img
    r,g,b = split(img)
    
    r = r[:,:] + ((r[:,:]/255) * amt)
    g = g[:,:] + ((g[:,:]/255) * amt)
    b = b[:,:] + ((b[:,:]/255) * amt)

    r,g,b = clip(r,g,b)
    r,g,b =to_uint8(r,g,b)
    
    output=output.astype('uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
            
    cv2.putText(output,"Exposure",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

def contrast_image_broken(img, contrast):
    test = img
    test=test.astype('float64')
    for rgb in range(0,2):
        test[:,:,rgb] = (img[:,:,rgb]- 128) * contrast + 128
        test = np.clip(test,0,255)
        
    test = test.astype('uint8')
    return test
    
def contrast_image(img,contrast):
    output = img
    r,g,b = split(output)
    b=b.astype('float64')
    g=g.astype('float64')
    r=r.astype('float64')

    r = (r-128) * contrast + 128
    g = (g-128) * contrast + 128
    b = (b-128) * contrast + 128
    
    r,g,b = clip(r,g,b)
    r,g,b = to_uint8(r,g,b)
     
    output=output.astype('uint8')

    for i in range(img.shape[0]):   
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    
    output = np.clip(output,0,255)
    output=output.astype('uint8')
    cv2.putText(output,"Contrast",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

def image_temp(img, amt):
    
    r,g,b = split(img)
    b=b.astype('float32')
    r=r.astype('float32')
    
    b = b[:,:] - amt
    r = r[:,:] + amt
    
    b = np.clip(b,0,255)
    r = np.clip(r,0,255)
    
    r,g,b =to_uint8(r,g,b)

    output = np.zeros((img.shape[0],img.shape[1],3))
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    cv2.putText(output,"Temperature",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

def image_tint(img, amt):
    
    r,g,b = split(img)
    g=g.astype('float32')
    
    g = g[:,:] + amt
    
    g = np.clip(g,0,255)
    
    g=g.astype('uint8')

    output = np.zeros((img.shape[0],img.shape[1],3))
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    cv2.putText(output,"Tint",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

def greyscale(img):
    r,g,b = split(img)

    b = b/255
    g = g/255
    r = r/255
    
    #Gamma expansion 
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if(r[i,j] < 0.04045): r[i,j] = r[i,j]/12.92
            if(b[i,j] < 0.04045): b[i,j] = b[i,j]/12.92
            if(g[i,j] < 0.04045): g[i,j] = g[i,j]/12.92
    r = ((r+0.055)/1.055)**2.4
    g = ((g+0.055)/1.055)**2.4
    b = ((b+0.055)/1.055)**2.4
    
    Y = 0.2126*r + 0.7152*g + 0.0722*b
    
    #Gamma Compression
    output = np.zeros((img.shape[0],img.shape[1],1))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if Y[i,j] <= 0.0031308:
                output[i,j] = 12.92*Y[i,j]
            else:
                output[i,j] = (1.055*(Y[i,j]**(1/2.4)))-0.055
    
    output = output*255        
    output = np.clip(output,0,255)
    output = output.astype('uint8')
    cv2.putText(output,"Greyscale",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
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

def threshold(img, amt):
    greyed = greyscale(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(greyed[i,j]<amt):greyed[i,j] = 0
            else: greyed[i,j] = 255
    return greyscale_to_rgb(greyed)

def color_invert(img):
    output = np.zeros((img.shape[0],img.shape[1],3))
    r,g,b = split(img)
    r = r*(-1) + 255
    g = g*(-1) + 255
    b = b*(-1) + 255
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
                
    output = output.astype('uint8')
    cv2.putText(output,"Inverted Color",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

def solarize(img, threshold):
    output = np.zeros((img.shape[0],img.shape[1],3))
    output = deepcopy(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for rgb in range(0,3):
                pix = output[i,j,rgb]
                if(pix > threshold):
                    pix = pix*(-1) + 255
                    pix = pix.astype('uint8')
                    output[i,j, rgb] = pix
                
    output = output.astype('uint8')
    cv2.putText(output,"Solarized",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

def image_flip_horizontal(img):
    output = img
    r,g,b = split(img)
    output=output.astype('uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = b[i,img.shape[1]-1-j], g[i,img.shape[1]-1-j], r[i,img.shape[1]-1-j]
    cv2.putText(output,"Flip Horizontal",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

def image_flip_vertical(img):
    output = img
    r,g,b = split(img)
    output=output.astype('uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = b[img.shape[0]-1-i,j], g[img.shape[0]-1-i,j], r[img.shape[0]-1-i,j]
    cv2.putText(output,"Flip Vertical",position,cv2.FONT_HERSHEY_PLAIN ,1,(0, 0, 255, 255),1)
    return output

##Part 2
def box_blur(image, kernel_size):
    kernel = np.zeros((kernel_size,kernel_size,3))
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    kernel[:] = 1
    kernel = kernel/(kernel_size**2)
    
    for i in range(0, image.shape[0],1):
        for j in range(0, image.shape[1],1):    
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            k1 = (roi * kernel)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            output[i][j]=k3
            
    output=output.astype('uint8')
    return output

def scale_image(img, percentage: float):
    width = int(img.shape[1] * percentage)
    height = int(img.shape[0] * percentage)
    dim = (width, height)
    return cv2.resize(img,dim,interpolation = cv2.INTER_NEAREST)

def gaussian_blur(image, kernel_size,):
    #kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    kernel = gaussian_kernel(kernel_size, sigma=3)
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
    #show_kernel(kernel[:,:,0],"kernel")
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
    #show_kernel(kernel,"kernel")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            #show_region(roi[:,:,0],"ROI")
            k1 = (kernel * roi)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            k3 = np.clip(k3,0,255)
            output[i][j]=k3
    output = np.clip(output,0,255)
    output = output.astype('uint8')
    return output

def convolution_grey(image, kernel):
    output = np.zeros(image.shape)
    padded_image = pad_image(image, kernel)
    #show_kernel(kernel,"kernel")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1],:]
            #show_region(roi[:,:],"ROI")
            k1 = kernel * roi
            k4 = abs(np.sum(k1))
            k4 = np.clip(k4,0,255)
            output[i,j] = k4 
        #cv2.imshow("Test", output)
    return output

def sharpening_filter(image,amt):
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
    kernel1 = show_kernel(kernel[:,:,0], "Kernel")
    #show_kernel(kernel[:,:,0],"kernel")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel_width, j:j + kernel_height,:]
            roi1 = show_region(roi, "ROI")
            centerpix = roi[kernel_width//2,kernel_height//2,:]
            centerpixRGB = np.sum(centerpix, axis = 0)
            kernel = gaussian_kernel(kernel_size,3)
            kernelsum1 = np.sum(kernel)
            
            
            for k in range(kernel_width):
                for l in range(kernel_height):
                    kernelpixRGBsum = np.sum(roi[k,l], axis = 0)      
                    diff = 0.0
                    diff = abs(kernelpixRGBsum.astype('float64') - centerpixRGB.astype('float64'))
                    if diff > 40 :
                        kernel [k,l] = 0

            kernelsum2 = np.sum(kernel)

            ##normalize kernel
            for rgb in range(0,3):
                sum = np.sum(kernel[:,:,rgb])
                for k in range(kernel_width):
                    for l in range(kernel_width): 
                        kernel[k,l,rgb] /= sum

            kernel1 = show_kernel(kernel[:,:,0], "Kernel")
            kernelsum2 = np.sum(kernel)
            k1 = (kernel * roi)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            k3 = np.clip(k3,0,255)
            output[i][j][:]=k3
        #show_progress(output,"TempOutput")
        
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
            roi=roi.astype('float64')
            roi = roi**2
            for k in range(0,3):
                pixel = np.median(roi[:,:,k])
                pixel = np.sqrt(pixel)
                pixel = np.clip(pixel,0,255)
                output[i,j,k]=pixel

    output=output.astype('uint8')
    return output

def gaussian_kernel(size, sigma=3):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
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
    img = scale_image(img,20)
    cv2.imshow(title, img)
    return img

def show_region(region, title):
    region = scale_image(region,15)
    region=region.astype('uint8')
    cv2.imshow(title, region)
    return region
    
def show_progress(region, title):
    region = scale_image(region,10)
    region=region.astype('uint8')
    cv2.imshow(title, region)
    
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
    padded_image = padded_image.astype('uint8')
    return padded_image

def subsample_image(image,kernelsize = 2):
    kernel = np.zeros((kernelsize,kernelsize,3))
    output = np.zeros((int(image.shape[0]/2)+1,int(image.shape[1]/2)+1,3))
    padded_image = pad_image(image, kernel)
    for i in range(0,image.shape[0],kernelsize):
        for j in range(0,image.shape[1],kernelsize):
            roi = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            roi=roi.astype('float64')
            roi = roi**2
            for k in range(0,3):
                pixel = np.median(roi[:,:,k])
                pixel = np.sqrt(pixel)
                pixel = np.clip(pixel,0,255)
                output[int(i//2),int(j//2),k]=pixel

    output=output.astype('uint8')
    return output

def gaussian_pyramid(image):
    image = gaussian_blur(image,3)
    image = subsample_image(image)
    return image

def gaussian_pyramid_bilateral(image):
    image = bilateral_filter(image,3)
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

def make_same_size(image1, image2):
    width = 0
    height = 0
    if image1.shape[0]<image2.shape[0] :
        width = image2.shape[0]
    else: width = image1.shape[0]
    if image1.shape[1]<image2.shape[1] :
        height = image2.shape[1]
    else: height = image1.shape[1]
    image1mod = np.zeros((width,height,3))
    image2mod = np.zeros((width,height,3))
    for i in range(width):
        for j in range(height):
            image1mod[i,j] = image1[i,j]
            image2mod[i,j] = image2[i,j]
    return image1mod, image2mod
    
def gaussian_pyramid(image):
    blurred = gaussian_blur(image,3)
    output = subsample_image(blurred,2)
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

    #show_kernel(kernel[:,:,0],"kernel")
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
                
                #show_kernel(kernel[:,:,rgb], "Kernel")
                #kernel[:,:,rgb] = normalize(kernel[:,:,rgb],0,1)
                k1 = (kernel[:,:,rgb] * roi[:,:,rgb])
                k2 = np.sum(k1, axis = 0)
                k3 = np.sum(k2, axis = 0)
                k3 = np.clip(k3,0,255)
                output[i,j,rgb]=k3
            #show_region(roi, "ROI")
        #show_progress(output,"TempOutput")
    return output

def unsharp_mask(image):
    r,g,b = split(image)
    blurred = gaussian_blur(image,9)
    r1,g1,b1 = split(blurred)
    r1 = r1.astype('float64')
    g1 = g1.astype('float64')
    b1 = b1.astype('float64') 
    maskR = r-r1
    maskG = g-g1
    maskB = b-b1
    r = r+maskR
    r = b+maskB
    r = g+maskG
    
    output  = np.zeros((image.shape[0],image.shape[1],3))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    output = np.clip(output,0,255)
    output = output.astype('uint8')
    return output