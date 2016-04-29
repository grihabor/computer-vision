from numpy import zeros
import numpy as np
import math

def seam_carve(img, mode, mask=None):

    mode = mode.split(' ')

    if mask is None:
        mask = zeros(img.shape[0:2])

    #calculate Y component
    Y = zeros(img.shape[0:2])
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            Y[y, x] = (0.299*img[y, x, 0] + 0.587*img[y, x, 1] + 0.114*img[y, x, 2])

    #calculate gradient
    grad = zeros(img.shape[0:2])
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            dx = 0
            if x > 0 and x + 1 < img.shape[1]:
                dx = Y[y, x + 1] - Y[y, x - 1]
            
            dy = 0
            if y > 0 and y + 1 < img.shape[0]:
                dy = Y[y + 1, x] - Y[y - 1, x]
            grad[y, x] = math.sqrt(dx*dx + dy*dy)
    
    
    
    scale = img.shape[0]*img.shape[1]*256
    
    if mode[0] == 'horizontal':
        #calculate energy matrix
        energy = zeros(img.shape[0:2])        
        energy[0, :] = grad[0, :] + mask[0, :] * scale
        for y in range(1, img.shape[0]):
            for x in range(0, img.shape[1]):
                min_energy = energy[y - 1, x]
                if x - 1 >= 0:
                    if energy[y - 1, x - 1] < min_energy:
                        min_energy = energy[y - 1, x - 1]
                if x + 1 < img.shape[1]:
                    if energy[y - 1, x + 1] < min_energy:
                        min_energy = energy[y - 1, x + 1]
                energy[y, x] = grad[y, x] + min_energy + mask[y, x] * scale
        
        
        if mode[1] == 'shrink': 
            #initialize
            resized_shape = (img.shape[0], img.shape[1] - 1, img.shape[2])
            resized_img = np.zeros(resized_shape)
            resized_mask = np.zeros(resized_shape[:2])
            
            carve_mask = np.zeros(energy.shape)
            index = np.argmin(energy[img.shape[0]-1, :])
            for y in range(img.shape[0] - 1, -1, -1):
                carve_mask[y, index] = 1
                
                #delete pixel from line
                resized_img[y, :index, :] = img[y, :index, :]
                resized_img[y, index:, :] = img[y, index + 1:, :]
                
                
                resized_mask[y, :index] = mask[y, :index]
                resized_mask[y, index:] = mask[y, index + 1:]
                
                if y == 0:
                    break
                    
                    
                ind = index
                min_energy = energy[y - 1, index]
                
                if index - 1 >= 0:
                    if energy[y - 1, index - 1] <= min_energy:
                        min_energy = energy[y - 1, index - 1]
                        ind = index - 1
                                               
                if index + 1 < img.shape[1]:
                    if energy[y - 1, index + 1] < min_energy:
                        min_energy = energy[y - 1, index + 1]
                        ind = index + 1   
                index = ind
        
        
############################################################
        
        
        
        elif mode[1] == 'expand': 
            #initialize
            resized_shape = (img.shape[0], img.shape[1] + 1, img.shape[2])
            resized_img = np.zeros(resized_shape)
            resized_mask = np.zeros(resized_shape[:2])
            
            carve_mask = np.zeros(energy.shape)
            index = np.argmin(energy[img.shape[0]-1, :])
            for y in range(img.shape[0] - 1, -1, -1):
                carve_mask[y, index] = 1
                
                #add pixel
                resized_img[y, :index + 1, :] = img[y, :index + 1, :]
                if index + 1 < img.shape[1]:
                    resized_img[y, index + 1, :] = (img[y, index, :]/2. + img[y, index + 1, :]/2.)
                else:                    
                    resized_img[y, index + 1, :] = img[y, index, :]
                resized_img[y, index + 2:, :] = img[y, index + 1:, :]
                
                
                resized_mask[y, :index + 1] = mask[y, :index + 1]
                resized_mask[y, index + 1] = 1
                resized_mask[y, index + 2:] = mask[y, index + 1:]
                
                if y == 0:
                    break
                    
                    
                ind = index
                min_energy = energy[y - 1, index]
                
                if index - 1 >= 0:
                    if energy[y - 1, index - 1] <= min_energy:
                        min_energy = energy[y - 1, index - 1]
                        ind = index - 1
                                               
                if index + 1 < img.shape[1]:
                    if energy[y - 1, index + 1] < min_energy:
                        min_energy = energy[y - 1, index + 1]
                        ind = index + 1   
                index = ind    
                
                
############################################################
        

    #sorry for copy&paste
    elif mode[0] == 'vertical':# shrink':
        #calculate energy matrix
        energy = zeros(img.shape[0:2])
        energy[:, 0] = grad[:, 0] + mask[:, 0] * scale
        for x in range(1, img.shape[1]):
            for y in range(0, img.shape[0]):
                min_energy = energy[y, x - 1]
                if y - 1 >= 0:
                    if energy[y - 1, x - 1] < min_energy:
                        min_energy = energy[y - 1, x - 1]
                if y + 1 < img.shape[0]:
                    if energy[y + 1, x - 1] < min_energy:
                        min_energy = energy[y + 1, x - 1]
                energy[y, x] = grad[y, x] + min_energy + mask[y, x] * scale
            
        if mode[1] == 'shrink':    
            #initialize 
            resized_shape = (img.shape[0]-1, img.shape[1], img.shape[2])
            resized_img = np.zeros(resized_shape)
            resized_mask = np.zeros(resized_shape[:2])
            
            carve_mask = np.zeros(energy.shape)
            index = np.argmin(energy[:, img.shape[1]-1])
            
            for x in range(img.shape[1] - 1, -1, -1):
                carve_mask[index, x] = 1
                
                #delete pixel from line
                resized_img[:index, x, :] = img[:index, x, :]
                resized_img[index:, x, :] = img[index + 1:, x, :]
                
                resized_mask[:index, x] = mask[:index, x]
                resized_mask[index:, x] = mask[index + 1:, x]
                
                if x == 0:
                    break
                min_energy = energy[index, x - 1]
                ind = index
                if index - 1 >= 0:
                    if energy[index - 1, x - 1] <= min_energy:
                        min_energy = energy[index - 1, x - 1]
                        ind = index - 1
                if index + 1 < img.shape[0]:
                    if energy[index + 1, x - 1] < min_energy:
                        min_energy = energy[index + 1, x - 1]
                        ind = index + 1       
                index = ind
            
            
############################################################
         
        elif mode[1] == 'expand':    
            #initialize 
            resized_shape = (img.shape[0]+1, img.shape[1], img.shape[2])
            resized_img = np.zeros(resized_shape)
            resized_mask = np.zeros(resized_shape[:2])
            
            carve_mask = np.zeros(energy.shape)
            index = np.argmin(energy[:, img.shape[1]-1])
            
            for x in range(img.shape[1] - 1, -1, -1):
                carve_mask[index, x] = 1
                
                #add pixel
                resized_img[:index + 1, x, :] = img[:index + 1, x, :]
                if index + 1 < img.shape[0]:
                    resized_img[index + 1, x, :] = (img[index, x, :]/2. + img[index + 1, x, :]/2.)
                else:
                    resized_img[index + 1, x, :] = img[index, x, :]
                resized_img[index + 2:, x, :] = img[index + 1:, x, :]
                
                resized_mask[:index + 1, x] = mask[:index + 1, x]
                resized_mask[index + 1, x] = 1
                resized_mask[index + 2:, x] = mask[index + 1:, x]
                
                if x == 0:
                    break
                min_energy = energy[index, x - 1]
                ind = index
                if index - 1 >= 0:
                    if energy[index - 1, x - 1] <= min_energy:
                        min_energy = energy[index - 1, x - 1]
                        ind = index - 1
                if index + 1 < img.shape[0]:
                    if energy[index + 1, x - 1] < min_energy:
                        min_energy = energy[index + 1, x - 1]
                        ind = index + 1       
                index = ind
            
    '''
    img[:, :, 0] = grad[:, :]# * 255
    img[:, :, 1] = grad[:, :]# * 255
    img[:, :, 2] = grad[:, :]# * 255
    '''
    return (resized_img, resized_mask, carve_mask)

