import numpy as np
#from skimage.filters import *
import math
import skimage.exposure as exp
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
import skimage.transform as trans
from skimage.feature import hog
from random import randint

#import matplotlib.pyplot as plt 
        
        
#add paddings to detect
#check 
        
        
edge_px = 16
width = 64
height = 128

nBins = 18
cell_size = (6, 6)
block_size = (3, 3)

blocks_in_column = height // cell_size[0] - block_size[0] + 1
blocks_in_row    = width  // cell_size[1] - block_size[1] + 1

SHOW_PLOTS = False
FAST_TEST = False
DEBUG_PRINT = False
bbox_width = 2

scale_from = 1.34
#we don't care about scale_to
scale_to = 100
scale_step = .33

scale_size_range = np.arange(scale_from, scale_to, scale_step)
scale_width_range = np.arange(0.7, 1.51, 0.1)

block_step = 5

num_roi = 12
prob_threshold = .75

def calc_roi(x, y, scale, prob, image_pad):
    roi = np.zeros((5))
    roi[0] = int(scale[1] * (x*cell_size[1] + edge_px)          - image_pad[1])
    roi[1] = int(scale[0] * (y*cell_size[0] + edge_px)          - image_pad[0])
    roi[2] = int(scale[1] * (x*cell_size[1] + width - edge_px)  - image_pad[1])
    roi[3] = int(scale[0] * (y*cell_size[0] + height - edge_px) - image_pad[0]) 
    roi[4] = prob
    return roi

def roi_ok(roi, img_shape):
    '''
    print(roi)
    print(img_shape)
    '''
    if roi[0] < 0 or roi[1] < 0:
        return False
    if roi[2] >= img_shape[1] or roi[3] >= img_shape[0]:
        return False
    return True
    
def add_image_padding(img, padding):        
    return np.pad(img, pad_width = ((padding[0], padding[0]), (padding[1], padding[1])), mode='edge')
    
def isintersect(rectbb,rectgt,threshold=0.5):
    (x_gt_from, y_gt_from, x_gt_to, y_gt_to) = rectgt[:4]
    (x_bb_from, y_bb_from, x_bb_to, y_bb_to) = rectbb[:4]
    if (min(x_bb_to, x_gt_to) <= max(x_bb_from, x_gt_from)) or \
            (min(y_bb_to, y_gt_to) <= max(y_bb_from, y_gt_from)):
        return False
        
    intersection = \
        (min(x_bb_to, x_gt_to) - max(x_bb_from, x_gt_from)) * \
        (min(y_bb_to, y_gt_to) - max(y_bb_from, y_gt_from))

    union = \
        (x_bb_to - x_bb_from) * (y_bb_to - y_bb_from) + \
        (x_gt_to - x_gt_from) * (y_gt_to - y_gt_from) - intersection

    return (intersection / float(union) >= threshold)

def remove_same_rois(roi_arr):
    skip_list = []
    for i in range(len(roi_arr)):
        for j in range(i+1, len(roi_arr)):
            if isintersect(roi_arr[i], roi_arr[j]):
                if roi_arr[i][4] > roi_arr[j][4]:
                    skip_list.append(j)
                else:
                    skip_list.append(i)
    result_arr = []
    for i in range(len(roi_arr)):
        if not(i in skip_list):
            result_arr.append(roi_arr[i])
    return np.array(result_arr)
   
    
#should return [x0; y0; x1; y1; measure] be careful!!
def detect(model, image, hogs=False, remove_duplicates=True, check_roi_arr=None):

    prob_arr = np.zeros((0))

    roi_arr = np.zeros((0, 5))
    hog_arr = None
    
    img_int = calc_intensity(image)     
    image_pad = [img_int.shape[0]//6, img_int.shape[1]//6]
    img_int = add_image_padding(img_int, image_pad)    
    '''
    plt.imshow(img_int, cmap='gray', interpolation='none')
    plt.show()
    '''
    break_flag = False
    
    if DEBUG_PRINT:
        print('img_int.shape = ', img_int.shape)
    
    for scale_size in scale_size_range:   
        if break_flag:
            break
        for scale_width in scale_width_range:
            
            scale = (scale_size, scale_size * scale_width)
                    
            scaled_shape = (img_int.shape[0]//scale[0], img_int.shape[1]//scale[1])
            resized_shape = (scaled_shape[0] - (scaled_shape[0] % cell_size[0]), \
                            scaled_shape[1] - (scaled_shape[1] % cell_size[1]))

            if resized_shape[0] < height or resized_shape[1] < width:
                if DEBUG_PRINT:
                    print('resized_shape ', resized_shape)
                    print('break')
                break_flag = True
                break
            img_resized = trans.resize(img_int, resized_shape)
            
            image_block_hog = hog(img_resized, \
                orientations=nBins, \
                pixels_per_cell=cell_size, \
                cells_per_block=block_size,\
                feature_vector=False)
                
            #print('img_resized.shape = ', img_resized.shape)
            #print(image_block_hog.shape)
            
            desc = None
            for y in range(0, image_block_hog.shape[0] - blocks_in_column, block_step):
                for x in range(0, image_block_hog.shape[1] - blocks_in_row, block_step):
                    
                    if(desc is None):
                        desc = image_block_hog[y:y+blocks_in_column, x:x+blocks_in_row, :, :, :]
                    else:
                        desc[:, :, :, :, :] = image_block_hog[y:y+blocks_in_column, x:x+blocks_in_row, :, :, :]
                    desc_flat = desc.ravel()
                    proba = model.predict_proba([desc_flat])[0]
                    
                    if proba[1] > prob_threshold:
                        roi_to_add = calc_roi(x, y, scale, proba[1], image_pad)
                        if roi_ok(roi_to_add, image.shape):
                            if len(prob_arr) < num_roi:
                                #print('scale: ', scale)
                                roi_arr = np.append(roi_arr, [roi_to_add], axis=0)
                                
                                if hog_arr is None:
                                    hog_arr = np.array([desc_flat])
                                else:
                                    hog_arr = np.append(hog_arr, [desc_flat], axis=0)
                                prob_arr = np.append(prob_arr, proba[1])
                                
                            elif not np.all(prob_arr > proba[1]):
                                j = np.argmin(prob_arr)
                                roi_arr[j] = roi_to_add
                                hog_arr[j] = desc_flat
                                prob_arr[j] = proba[1]
                       
                #print(desc.shape)
                #print(y, ' of ', image_block_hog.shape[0] - blocks_in_column) 
               
    if remove_duplicates:
        if DEBUG_PRINT:
            print('rois before:')
            print(roi_arr)
        roi_arr = remove_same_rois(roi_arr)
        if DEBUG_PRINT:
            print('rois after:')
            print(roi_arr)
    
    '''
    #some test going on here
    if check_roi_arr is None:
        #print('roi_arr: ', roi_arr)  
        img = np.copy(image)
        for roi in roi_arr:
            img = add_bbox_around_roi(img, roi, [0, 0, 1])  
        if check_roi_arr is not None:
            print('check_roi_arr: ', check_roi_arr)
            for roi in check_roi_arr:
                image = add_bbox_around_roi(image, roi, [0, 1, 0])  
            
        plt.subplot(121)
        plt.imshow(image, interpolation='none')
        plt.subplot(122)
        plt.imshow(img, interpolation='none')
        plt.show()
    '''
        
        
    if hogs:
        return roi_arr, hog_arr
    return roi_arr
    
    

def add_bbox_around_roi(img, roi, color):
    img[roi[1]:roi[3], roi[0]-bbox_width:roi[0]] = color
    img[roi[1]:roi[3], roi[2]:roi[2]+bbox_width] = color
    img[roi[1]-bbox_width:roi[1], roi[0]:roi[2]] = color
    img[roi[3]:roi[3]+bbox_width, roi[0]:roi[2]] = color
    return img

def calc_intensity(img):
    img_int = 0.299 * img[:,:,0] + \
              0.587 * img[:,:,1] + \
              0.114 * img[:,:,2]
    return img_int
    
def add_bbox_paddings(roi):
    bbox_pad_ver = (roi[2] - roi[0]) * edge_px // (height - 2 * edge_px)
    bbox_pad_hor = (roi[3] - roi[1]) * edge_px // (width - 2 * edge_px)
    return [roi[0] - bbox_pad_ver, roi[1] - bbox_pad_hor, roi[2] + bbox_pad_ver, roi[3] + bbox_pad_hor]
    
def calc_hog_desc(img):
    return hog(img, orientations=18, pixels_per_cell=(6, 6), cells_per_block=(3, 3))
    
def train_clf(features_arr, labels_arr, linear=False):
    if DEBUG_PRINT:
        print('svm start')
    if linear:
        kernel = 'linear'
    else:
        kernel = 'rbf'
    tmp = svm.SVC(kernel=kernel, C=1e3, gamma=0.03, probability=True)
    clf = tmp.fit(features_arr, labels_arr)
    if DEBUG_PRINT:
        print('svm done')
    return clf
    
def train_detector(imgs, gt):
    #images = np.copy(imgs)
    features_arr = None
    labels_arr = np.zeros((0), dtype=np.int)
    no_human_ind_arr = np.zeros((0), dtype=np.int)
    
    for i in range(0, len(gt)):
        bbox_arr = gt[i]
        cur_img = imgs[i]
        
        if FAST_TEST:
            if i > 50:
                break
        
        
        img_int = calc_intensity(cur_img)
        image_pad = [img_int.shape[0]//4, img_int.shape[1]//4]
        img_int = add_image_padding(img_int, image_pad)        
        
        '''
        print(i, '/', len(gt))
        plt.imshow(img_int, cmap='gray',interpolation='none')
        plt.show()
        '''
        
        '''
        in case there are no people
        save the index to use later
        '''
        if len(bbox_arr) == 0:
            no_human_ind_arr = np.append(no_human_ind_arr, i)        
        
        j = 2
        '''
        if there are people
        '''
        for roi in bbox_arr:
            #bbox is [top, left, bottom, right] - be careful!!
            '''
            cur_img = add_bbox_around_roi(cur_img, roi, [1, 0, 0])
            '''
            roi = [roi[1]+image_pad[0], roi[0]+image_pad[1], 
                   roi[3]+image_pad[0], roi[2]+image_pad[1]]
            
            #add edge_px padding
            roi = add_bbox_paddings(roi)            
            image_to_hog = img_int[roi[0] : roi[2], roi[1] : roi[3]]
            
            #resize to (height, width)                       
            image_to_hog = trans.resize(image_to_hog, (height, width))
            
            
            if SHOW_PLOTS:
                plt.subplot(2, 4, min(4, j))
                plt.imshow(image_to_hog,cmap='gray', interpolation='none')
                plt.subplot(2, 4, 4+min(4, j))
                plt.imshow(np.fliplr(image_to_hog),cmap='gray', interpolation='none')
                j += 1
            
            
            hog_desc = [calc_hog_desc(image_to_hog), calc_hog_desc(np.fliplr(image_to_hog))]
            
            #print(hog_desc)
            
            if features_arr is None:
                features_arr = np.zeros((0, hog_desc[0].shape[0]))
            
            features_arr = np.append(features_arr, hog_desc, axis=0)
            labels_arr = np.append(labels_arr, 1)    
            labels_arr = np.append(labels_arr, 1)    
        
        
        if SHOW_PLOTS:# or labels_arr[-1] == 0:
            plt.subplot(1, 4, 1)    
            plt.imshow(cur_img, interpolation='none')
            plt.show()
        
    num_humans = len(labels_arr)
    if DEBUG_PRINT:
        print('num_humans: ', num_humans)
    n = 5 * len(labels_arr) // len(no_human_ind_arr)
    if DEBUG_PRINT:
        print('need to find non humans: ', n)
        print(no_human_ind_arr)
    for index in no_human_ind_arr:
        cur_img = imgs[index]
        img_int = calc_intensity(cur_img)
                  
        for i in range(n):
            roi = np.zeros((4))
            roi[0] = randint(0, cur_img.shape[0]-height-1)
            roi[1] = randint(0, cur_img.shape[1]-width-1)
            roi[2] = roi[0] + randint(height, cur_img.shape[0]-roi[0]) 
            roi[3] = roi[1] + randint(width, cur_img.shape[0]-roi[0])
            image_to_hog = img_int[roi[0]:roi[2], roi[1]:roi[3]]
            image_to_hog = trans.resize(image_to_hog, (height, width))
            '''
            plt.imshow(image_to_hog,cmap='gray', interpolation='none')
            plt.show()
            '''
            desc = calc_hog_desc(image_to_hog)
            
            if features_arr is None:
                features_arr = np.zeros((0, desc.shape[0]))
            
            features_arr = np.append(features_arr, [desc], axis=0)
            labels_arr = np.append(labels_arr, 0)
    
    num_non_humans = len(labels_arr) - num_humans
    if DEBUG_PRINT:
        print('num_non_humans: ', num_non_humans)
        print('features: ', features_arr.shape)
    
    clf = train_clf(features_arr, labels_arr, linear=True)
    
    stop_threshold = num_humans + int(1.1 * num_non_humans)
    if DEBUG_PRINT:
        print('stop_threshold = ', stop_threshold)
    
    
    for i in range(0, len(gt)):           
        bbox_arr = gt[i]
        cur_img = imgs[i]
        
        rect_arr, hog_arr = detect(clf, cur_img, hogs=True, remove_duplicates=False, check_roi_arr = bbox_arr)
        if DEBUG_PRINT:
            print('rect_arr.shape = ', rect_arr.shape)
        if rect_arr.shape[0] > 0:
            if DEBUG_PRINT:
                print(hog_arr.shape)
            for i, detected_rect in enumerate(rect_arr):
                inters = False
                for bbox in bbox_arr:
                    #print('bbox ', bbox.shape, ' ', detected_rect.shape)
                    if isintersect(bbox, detected_rect[:4]):
                        inters = True
                        break
                if not inters:
                    if len(labels_arr) < stop_threshold:
                        features_arr = np.append(features_arr, [hog_arr[i]], axis=0)
                        labels_arr = np.append(labels_arr, [0])
                    else:
                        break
                 
            if DEBUG_PRINT:       
                print('labels/labels_max: ', len(labels_arr), '/', stop_threshold)
                print('features: ', features_arr.shape)
            '''
            clf = train_clf(features_arr, labels_arr)
            '''
        if FAST_TEST or len(labels_arr) >= stop_threshold:
            break
        
    clf = train_clf(features_arr, labels_arr)
        
        
    return clf
