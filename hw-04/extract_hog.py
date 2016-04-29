import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color

from skimage.filters import sobel_h, sobel_v
import math
import skimage.exposure as exp
import skimage.transform as trans

#8 3 1 0       0.845793
#8 4 2 .1     0.857793
#8 4 2 0     0.863724

#6 4 2 0      0.84
#12 4 2 0     0.89

#8 4 2 0 from 0 to pi     0.863724



binCount = 10
numCells = 4
blockSize = 2
step = 2
edge_k = 0

numBlocks = int((numCells - blockSize + 1)/step)

def calc_hists(grad_dir, rect, cellRows, cellColumns, edge):
    
    hist_arr = np.zeros((0, numCells, binCount))
    for y in range(0, numCells):
        hist_row = np.zeros((0, binCount))
        for x in range(0, numCells):
            cell = grad_dir[rect[0] + cellRows * y - edge:rect[0] + cellRows * (y + 1) + edge, rect[1] + cellColumns * x - edge:rect[1] + cellColumns * (x + 1) + edge]
            hist,bin_edges = np.histogram(cell, bins=binCount, range=(-math.pi, math.pi))
            
            #plt.subplot(numCells, numCells+1, 1+ y*(numCells+1) + x)            
            #plt.bar(range(0,binCount), hist)
            
            hist.resize((1, hist.shape[0]))
            hist_row = np.append(hist_row, hist, axis=0)
        
        hist_row.resize((1, hist_row.shape[0], hist_row.shape[1]))
        #print(hist_row)
        hist_arr = np.append(hist_arr, hist_row, axis=0)
    
    return hist_arr   

from sklearn.preprocessing import normalize
    
def calc_block_hists(hist_arr):
    
    block_hist_arr = np.zeros((0, numBlocks, binCount))
    for y in range(0, step*numBlocks, step):
        block_hist_row = np.zeros((0, binCount))
        for x in range(0, step*numBlocks, step):
            block_hist = np.sum(hist_arr[y:y+blockSize, x:x+blockSize], axis=(0, 1))
            block_hist = normalize([block_hist])
            block_hist.resize((binCount))
            '''
            plt.subplot(numBlocks, 2*numBlocks, 1+y*(2*numBlocks) + x)        
            plt.bar(range(0,binCount), block_hist)
            plt.axis([0, binCount, 0, 1])
            '''
            block_hist.resize((1, block_hist.shape[0]))
            block_hist_row = np.append(block_hist_row, block_hist, axis=0)
        
        block_hist_row.resize((1, block_hist_row.shape[0], block_hist_row.shape[1]))
        block_hist_arr = np.append(block_hist_arr, block_hist_row, axis=0)
        
    return block_hist_arr
    

def extract_hog(img, roi):
    
    #plt.subplot(243)
    #plt.imshow(img, interpolation='none')
        
    corr_img = img
        
    img_int = 0.299 * corr_img[:,:,0] + \
              0.587 * corr_img[:,:,1] + \
              0.114 * corr_img[:,:,2]
    """
    #plt.subplot(244)
    #plt.hist(img_int)
    
    
    #plt.subplot(247)
    #plt.imshow(img_int, cmap='gray', interpolation='none')
    
    img_sob_y = sobel_h(img_int)
    img_sob_x = sobel_v(img_int)
    #grad_val = np.sqrt(img_sob_y**2 + img_sob_x**2)
    
    #plt.subplot(132)
    #plt.imshow(img_sobel, cmap='gray', interpolation='none')
    
    grad_dir = np.arctan2(img_sob_y, img_sob_x);
    '''
    for y in range(0, grad_dir.shape[0]):
        for x in range(0, grad_dir.shape[1]):
            if grad_dir[y, x] < 0:
                grad_dir[y, x] += np.pi   
    '''
        
    cellRows = int((roi[2] - roi[0]) / numCells + .5)
    cellColumns = int((roi[3] - roi[1]) / numCells + .5)
    
    rect = np.zeros((4), dtype=np.int)
    rect[0] = roi[0]
    rect[1] = roi[1]
    rect[2] = roi[0] + cellRows*numCells
    rect[3] = roi[1] + cellColumns*numCells
    
    edge = int(cellRows * edge_k)
    #print(cellRows, ' ', edge)
    
    hist_arr = calc_hists(grad_dir, rect, cellRows, cellColumns, edge)
    #plt.show()
    block_hist_arr = calc_block_hists(hist_arr)
       
    
    '''
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    '''
    #return block_hist_arr.flatten()
    """    
     
    #64 64, 16 16, 2 2
    #8 bins    0.9579   
    #10 bins   0.959034
    #12 bins   0.958483
    
    edge = int(img_int.shape[0] * 0.05)
    
    image = img_int[max(0, roi[0] - edge):min(img_int.shape[0]-1, roi[2] + edge), 
                    max(0, roi[1] - edge):min(img_int.shape[1]-1, roi[3] + edge)]
    image = trans.resize(image, (72, 72))
    
    fd = hog(image, orientations=10, pixels_per_cell=(18, 18), cells_per_block=(2, 2))
    
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exp.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
    '''
    return np.abs(fd)