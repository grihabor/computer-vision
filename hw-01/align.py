from numpy import array, dstack, roll
import numpy as np
from skimage.transform import rescale
import time

def MSE(im1, im2, offset = (0, 0)):
    shift = [0, 0]
    min_shift = [0, 0]
    minsum = -1.0
    cursum = 0.0
    for shift[0] in range(-1, 2):
        for shift[1] in range(-1, 2):
            cursum = 0.0
            height = im1.shape[0] - 2
            width = im1.shape[1] - 2
            
            for y in range(0, im1.shape[0] - 2):    
                yIndex = y + 1 - shift[0] + offset[0]
                if yIndex >= im2.shape[0]:
                    height = y
                    break
                    
                for x in range(0, im1.shape[1] - 2):    
                    xIndex = x + 1 - shift[1] + offset[1]
                    if xIndex >= im2.shape[1]:
                        width = x
                        break
                
                    cursum += (im1[y + 1][x + 1] - im2[yIndex][xIndex])**2
            
            cursum = cursum/width/height
            if (cursum < minsum or minsum < 0):
                minsum = cursum
                min_shift[:] = shift
                
    # second image is shifted by +min_shift
    # so to align images we need to 
    # subtract min_shift
    return [-min_shift[0], -min_shift[1]]
               
def CrossCorr(im1, im2, offset = (0,0)):
    shift = [0, 0]
    max_shift = [0, 0]
    maxcorr = 0.0
    curcorr = 0.0
    for shift[0] in range(-1, 2):
        for shift[1] in range(-1, 2):
            curcor = 0.0
            xsum = 0.0
            ysum = 0.0
            xysum = 0.0
            for y in range(0, im1.shape[0] - 2):    
                yIndex = y + 1 - shift[0] + offset[0]
                if yIndex >= im2.shape[0]:
                    break
                    
                for x in range(0, im1.shape[1] - 2):    
                    xIndex = x + 1 - shift[1] + offset[1]
                    if xIndex >= im2.shape[1]:
                        break
                
                    xsum += im1[y + 1][x + 1]
                    ysum += im2[yIndex][xIndex]
                    xysum  += im1[y + 1][x + 1]*im2[yIndex][xIndex]
            
            curcorr = xysum/(xsum*ysum)
            if (curcorr > maxcorr):
                maxcorr = curcorr
                max_shift[:] = shift
                
    return [-max_shift[0], -max_shift[1]]
              
def low_quality(input, n):
    image = array(input)
    for y in range(0, image.shape[0]-(n-1), n):
        for x in range(0, image.shape[1]-(n-1), n):
            image[y][x] = np.sum(image[y:y+n, x:x+n], axis=(0,1))/(n**2)
    image = image[::n, ::n]
    return image

def align(greyscale):
    start = time.time()
    # 1/3 of initial image
    shape = (int(greyscale.shape[0]/3), greyscale.shape[1], 3)
    # margin used for calculations
    margin = [int(shape[0]*0.10), int(shape[1]*0.10)]
    
    # get raw channels
    red   = [greyscale[             margin[0]:  shape[0] - margin[0], margin[1]:shape[1] - margin[1]]]
    green = [greyscale[  shape[0] + margin[0]:2*shape[0] - margin[0], margin[1]:shape[1] - margin[1]]]
    blue  = [greyscale[2*shape[0] + margin[0]:3*shape[0] - margin[0], margin[1]:shape[1] - margin[1]]]
       
    # first step of resizing (/ low if resize is big) 
    low = int(len(red[-1]) / 300 + 0.5)
    '''
    print("len = ", len(red[-1]))
    print("low = ", low)
    '''
    if low > 1:
        red.append(low_quality(red[-1], low))
        green.append(low_quality(green[-1], low))
        blue.append(low_quality(blue[-1], low))
        
    n = 0
    # resize while image is big
    while (len(red[-1])) > 30:
        red.append(low_quality(red[-1], 2))
        green.append(low_quality(green[-1], 2))
        blue.append(low_quality(blue[-1], 2))
        n=n+1
        
    k = 2**n    
    
    shG = [0, 0]
    shB = [0, 0]
    '''
    print("red len = ", len(red))
    print("n = ", n)
    '''
    # calculate shifts on each iteration
    for i in range(-1, -min(len(red)+1, 6), -1):
        shG = [2*shG[0], 2*shG[1]]
        shB = [2*shB[0], 2*shB[1]]        
        
        t = MSE(red[i], green[i], shG)
        shG = [shG[0] + t[0], shG[1] + t[1]]
        
        t = MSE(red[i], blue[i], shB)
        shB = [shB[0] + t[0], shB[1] + t[1]]
        '''
        print("sh green: ", shG)
        print("sh blue: ", shB)
        '''
        k=k/2
    
    '''
    k*=2
    print("k = ", k)    
    shG = [k*shG[0], k*shG[1]]
    shB = [k*shB[0], k*shB[1]]
    '''
    # undo the first resize step
    if low > 1:
        shG = [low*shG[0], low*shG[1]]
        shB = [low*shB[0], low*shB[1]]
       
    ''' 
    print()
    
    print("great sh green: ", shG)
    print("great sh blue: ", shB)
    '''
    
    margin = [int(shape[0]*0.07), int(shape[1]*0.07)]
        
    red   = greyscale[             margin[0]:  shape[0] - margin[0], margin[1]:shape[1] - margin[1]]   
    green = greyscale[  shape[0] + margin[0] + shG[0]:2*shape[0] - margin[0] + shG[0], margin[1] + shG[1]:shape[1] - margin[1] + shG[1]]
    blue  = greyscale[2*shape[0] + margin[0] + shB[0]:3*shape[0] - margin[0] + shB[0], margin[1] + shB[1]:shape[1] - margin[1] + shB[1]]
    
    """print(blue.shape, green.shape, red.shape)"""
    '''bgr_image = np.dstack([blue, green, red])'''
    
    bgr_image = np.zeros((red.shape[0], red.shape[1], 3))
    
    bgr_image[:,:,0] = blue[:,:]
    bgr_image[:,:,1] = green[:,:]
    bgr_image[:,:,2] = red[:,:]
            
    print("time total: ", time.time()-start)
    return bgr_image
