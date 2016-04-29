import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import *
import math
import skimage.exposure as exp

#8 4 2 - 0.83778
#8 5 2 - 0.83333
#8 5 3 - 0.80000
#8 3 2 - 0.75555
#8 6 4 - 0.77555
#8 6 3 - 0.83111

#8 3 1 0 - 0.85333
#8 3 1 .25 - 0.85333
#8 3 1 .1 - 0.86000


binCount = 8
numCells = 3
blockSize = 1
edge_k = .1

numBlocks = numCells + 1 - blockSize

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
    for y in range(0, numBlocks):
        block_hist_row = np.zeros((0, binCount))
        for x in range(0, numBlocks):
            block_hist = np.sum(hist_arr[y:y+blockSize, x:x+blockSize], axis=(0, 1))
            block_hist = normalize([block_hist])
            block_hist.resize((binCount))
            
            plt.subplot(numBlocks, 2*numBlocks, 1+y*(2*numBlocks) + x)        
            plt.bar(range(0,binCount), block_hist)
            plt.axis([0, binCount, 0, 1])
            
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
    return block_hist_arr.flatten()
