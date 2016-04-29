import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as exp
from skimage import filters
import skimage.morphology as morph
from skimage import measure
import skimage 
from skimage import transform
from skimage.morphology import reconstruction
import os 
from skimage.io import imread, imsave

def generate_template(digit_dir_path):
    
    template = None
    n = 0
    for (dirpath, dirnames, filenames) in os.walk(digit_dir_path):
        for filename in filenames:
            img = skimage.img_as_float(imread(digit_dir_path + '/' + filename, plugin='matplotlib'))
            if template is None:
                template = img
            else:            
                #max_height = argmax(template.shape[0], img.shape[0])
                #max_width = argmax(template.shape[1], img.shape[1])
                template = template + transform.resize(img, template.shape)
            n = n + 1
            
    template = template / n        
    #plt.imshow(template, cmap='gray', interpolation='none') 
    #plt.show()
                
    return template

'''
def draw(img, title, p21, p22, p23, p11=None, p12=None, p13=None):
    if not(p11 is None):
        hist, bins = exp.histogram(img, nbins=256)
        plt.subplot(p21, p12, p13)
        t = np.arange(0, 256)
        t = t[:hist.shape[0]]
        plt.plot(t, hist)    
    subplt = plt.subplot(p21, p22, p23)
    subplt.set_title(title)
    plt.imshow(img, cmap='gray', interpolation='none') 
    return
'''
    
def intensity_correction(img):
    mean_val = np.mean(img)
    mean_val = mean_val / 255.
    #print(mean_val)
    
    #sigmoid correction
    img2 = exp.adjust_sigmoid(img, cutoff = mean_val, gain=5)
    #draw(img2, "Sigmoid correction", 4, 3, 4, 4, 3, 5)
    
    #robust linear correction
    left, right = np.percentile(img2, (10, 90))
    img3 = exp.rescale_intensity(img2, in_range=(left, right))        
    #draw(img3, "Linear correction", 4, 3, 7, 4, 3, 8)
    
    img4 = filters.gaussian(img3, sigma=.5)    
    #draw(img4, "Gaussian filter", 4, 3, 10, 4, 3, 11)
    return img4
    
def make_empty_edge(img, edge):        
    big_img = np.zeros((img.shape[0] + 2*edge, img.shape[1] + 2*edge), dtype= np.int8)    
    big_img[edge:-edge, edge:-edge] = img[:,:]
    return big_img
    
    
def leave_the_biggest(img, num_regions=-1):
    labeled = measure.label(img, background=0)
    props = measure.regionprops(labeled)
    
    #choose the biggest area region and delete others
    areas = np.zeros((0))
    for i in range(0, len(props)):
        areas = np.append(areas, props[i].area)
    ind_max = np.argmax(areas)
    area = props[ind_max].area
    res_img = img
    if(num_regions == -1 or len(props) <= num_regions):
        props = props[ind_max]
        res_img = np.zeros(img.shape, dtype=np.int8)
        res_img[props.bbox[0]:props.bbox[2], props.bbox[1]:props.bbox[3]] = props.image.astype(np.int8)
    
    return (res_img, area)
    
    
        
def get_binary(img):
    
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.min()
    mask = img
    dilated = reconstruction(seed, mask, method='dilation')    
    mask_img = img - dilated
    mask_img = intensity_correction(mask_img)  
    #draw(mask_img, "mask image", 4, 3, 2)
    
    
    blsize = img.shape[0] / 3
    if blsize % 2 == 0:
        blsize += 1
    
    binary_img = filters.threshold_adaptive(img, block_size = blsize)
    binary_img = img > binary_img
    #draw(binary_img, "Binary image", 4, 3, 7) 
    
    h = filters.threshold_otsu(mask_img)
    binary_mask = mask_img > h
    #draw(binary_mask, "Binary mask image", 4, 3, 3) 
       
    msize = (img.shape[0] / 10) ** 2
    binary_mask = morph.remove_small_objects(binary_mask, min_size=msize)     
    #draw(binary_mask, "rm sm", 4, 3, 2) 
    
    #binary_mask, area = leave_the_biggest(binary_mask, num_regions = 2)
    
    edge = 50
    binary_mask = make_empty_edge(binary_mask, edge)
    
    dilsize = img.shape[0] / 3
    if dilsize % 2 == 0:
        dilsize += 1
        
    sel = morph.square(dilsize)
    binary_mask = morph.binary_dilation(binary_mask, selem = sel)
        
    #draw(binary_mask, "after dilation", 4, 3, 5) 
    binary_mask = morph.convex_hull_image(binary_mask)
    
    
    binary_mask, area = leave_the_biggest(binary_mask)
    
    #draw(binary_mask, "Binary mask", 4, 3, 6)     
    
    #binary_mask = morph.remove_small_holes(binary_mask, min_size = area)
    
    binary_mask = morph.binary_erosion(binary_mask, selem = sel)
    
    wide_sel = np.zeros((dilsize, dilsize))
    wide_sel[dilsize/2-1:dilsize/2+2,:] = 1
    
    
    end_sel = morph.square(img.shape[0] / 4)
    
    binary_mask = morph.binary_closing(binary_mask, selem = end_sel)
    binary_mask = morph.binary_opening(binary_mask, selem = end_sel)
    
    binary_mask = binary_mask.astype(np.int8)
    binary_mask, area = leave_the_biggest(binary_mask)
    
    #draw(binary_mask, "final mask", 4, 3, 8)   
    
    binary_mask = binary_mask[edge:-edge, edge:-edge]
    
    binary_img = binary_img * binary_mask
    #draw(binary_img, "result image", 4, 3, 9) 
    
                  
    s = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])
    binary_img = morph.binary_opening(binary_img, selem = s)
    #draw(binary_img, "binary_opening", 4, 3, 10) 
    
    size = (img.shape[0] / 12) ** 2
    binary_img = morph.remove_small_objects(binary_img, min_size=size)
    
    
    
    #draw(binary_img, "rm small", 4, 3, 11) 
    
    
    
    return (binary_img, binary_mask)
    
    

def recognize(img, digit_templates):

    #draw(img, "Initial image", 4, 3, 1)    
    corrected_img = intensity_correction(img)
    #draw(corrected_img, "Corrected image", 4, 3, 4) 
  
    binary_img, binary_mask = get_binary(corrected_img)
    
    labeled_img = measure.label(binary_img, background=0)
    #draw(labeled_img, "labeled", 4, 3, 12) 
        
    props = measure.regionprops(labeled_img)
    #print(len(props))
    
    main_rect = measure.regionprops(binary_mask)
    main_rect = main_rect[0].bbox
    #print("main rect = ", main_rect)
    k = 0.6
    center = main_rect[3]*k + (1-k)*main_rect[1]
    #print(center)
    final = np.zeros(labeled_img.shape)
    
    regions = np.zeros((0, 3))
    for i in range(0, len(props)):
        left = props[i].bbox[1]
        height = props[i].bbox[2] - props[i].bbox[0]
        width = props[i].bbox[3] - props[i].bbox[1]
        if height > width and left < center and height < 3*width:
            regions = np.append(regions, [[height, props[i].bbox[1], i]], axis=0)
            
            #final[props[i].bbox[0]:props[i].bbox[2], props[i].bbox[1]:props[i].bbox[3]] = 1.
            
    regions = regions[np.argsort(regions[:,0])]
    regions[::-1] = regions[:]
    
    #print(regions)
    if len(regions) > 3:
        regions = regions[:4]
        regions = regions[np.argsort(regions[:,1])]
            
        #additional check
        average_top_first = (props[int(regions[0][2])].bbox[0]+props[int(regions[1][2])].bbox[0]+props[int(regions[2][2])].bbox[0]) / 3
        last_disp = abs(average_top_first - props[int(regions[3][2])].bbox[0])
        average_top_last  = (props[int(regions[1][2])].bbox[0]+props[int(regions[2][2])].bbox[0]+props[int(regions[3][2])].bbox[0]) / 3
        first_disp = abs(average_top_last - props[int(regions[0][2])].bbox[0])
        #print(abs(first_disp - last_disp))
        if abs(first_disp - last_disp) > 1:
            if (first_disp >= last_disp):
                regions = regions[1:]
            else:
                regions = regions[:-1]
        else:
            #print(regions)
            regions = regions[np.argsort(regions[:,0])]
            regions[::-1] = regions[:]            
            regions = regions[:3]
    
    regions = regions[np.argsort(regions[:,1])]
    
    #print(regions)
    
    
    plt.clf()
    #draw(labeled_img, "labeled", 3, 2, 2) 
    #draw(corrected_img, "corrected", 3, 2, 1) 
    
    
    result = [0, 0, 0]
    for i in range(0, len(regions)):
        digit_mask = props[int(regions[i][2])].image
        #draw(digit_mask, "digit", 3, 3, 4 + i) 
        
        bbox = props[int(regions[i][2])].bbox
        digit_img = corrected_img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        digit_mask = 1. - digit_mask.astype(np.float)
        coef = .4
        digit_img = (1 - coef) * digit_img + coef * digit_mask
        
        
        index = 0
        min_mse = 0
        
        for j in range(0, 10):
            mse = measure.compare_mse(digit_img, transform.resize(digit_templates[j], digit_img.shape))
            if j == 0:
                min_mse = mse
            elif min_mse > mse:
                min_mse = mse
                index = j            
        #draw(digit_img, str(index), 3, 3, 7 + i) 
        result[i] = index
    
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    #plt.show()
    
    return (result[0], result[1], result[2])
