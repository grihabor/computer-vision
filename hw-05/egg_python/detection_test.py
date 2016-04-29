#!/usr/bin/python

import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from sys import argv, stdout, exit
from os.path import basename
from glob import glob
from skimage.io import imread
from numpy import loadtxt, zeros, ones, array, vstack, hstack
from detection import train_detector, detect


def load_gt(path):
    gt_filename = path + '/gt.txt'
    filenames = [basename(x) for x in glob(path + '/*.png')]

    gt = {}
    imgs = []
    for filename in filenames:
        gt[filename] = zeros([0, 4])
        imgs.append(imread(path + '/' + filename, plugin = 'matplotlib'))

    data = loadtxt(gt_filename, delimiter=';', skiprows=1, usecols=range(1,5))

    for i, line in enumerate(open(gt_filename).readlines()[1:]):
        filename = line[0:9]
        gt[filename] = vstack([gt[filename], data[i, :]])
    sorted_list = [gt[filename] for filename in filenames]
    return (imgs, sorted_list)

    
def isintersect(rectbb,rectgt):
    (x_gt_from, y_gt_from, x_gt_to, y_gt_to) = rectgt
    (x_bb_from, y_bb_from, x_bb_to, y_bb_to) = rectbb
    if (min(x_bb_to, x_gt_to) <= max(x_bb_from, x_gt_from)) or \
            (min(y_bb_to, y_gt_to) <= max(y_bb_from, y_gt_from)):
        return False
        
    intersection = \
        (min(x_bb_to, x_gt_to) - max(x_bb_from, x_gt_from)) * \
        (min(y_bb_to, y_gt_to) - max(y_bb_from, y_gt_from))

    union = \
        (x_bb_to - x_bb_from) * (y_bb_to - y_bb_from) + \
        (x_gt_to - x_gt_from) * (y_gt_to - y_gt_from) - intersection

    return (intersection / float(union) >= 0.5)
        
def plot_curve(precision,recall):
    # Plot Precision-Recall curve
    precision_scaled = array(precision)/100.
    recall_scaled = array(recall)/100.
    plt.clf()
    plt.plot(recall_scaled, precision_scaled, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.show()
    
def compute_auc(all_truths,all_measures,good_gt_count,plot=False):
    same_confidence=1e-5  
    barrier = 1.5
    
    changes = [(all_measures[i],all_truths[i]) for i in range(all_measures.shape[0])]
    changes.sort() # by default using first element as key, ascending

    # starting from threshold = above max confidence of detection:
    # so there are no single one detection accepted, so FP = TP = 0, FN = all gt count
    tp_value = 0
    tpfp_value = 0 # tp + fp
    
    precision = lambda: (100. * tp_value / tpfp_value) if tpfp_value != 0 else None
    recall = lambda: 100. * tp_value / good_gt_count
    
    # data of previous point to measure gap between sequent points
    prev_precision = precision()
    prev_recall = recall()
    precision_list = []
    recall_list = []
    auc_score = 0
    force_point = True # force first point to be written without gap checks
    current_idx = len(changes) - 1
    while current_idx >= 0:
        # we need to take into account all detections with close confidence
        init_confidence = changes[current_idx][0]
        while current_idx >= 0:
            confidence, tp_change = changes[current_idx]
            if abs(init_confidence - confidence) > same_confidence:
                break
            # changes in this confidence point
            tp_value += tp_change
            tpfp_value += 1
            current_idx -= 1
        # recalculating precision/recall
        curr_precision = precision()
        curr_recall = recall()
        # calculating points gap
        precision_diff = abs(prev_precision - curr_precision) if prev_precision else 1. - curr_precision
        recall_diff = curr_recall - prev_recall
        # checking gap (or last point)
        gap = max(precision_diff, recall_diff)
        if gap > barrier or current_idx == 0 or force_point:
            if force_point:
                # processing the first curve point
                force_point = False
                prolongation = curr_precision * curr_recall
                auc_score += prolongation
            else:
                auc_score += (0.5 * recall_diff * (prev_precision + curr_precision))
            if plot:
                precision_list.append(curr_precision)
                recall_list.append(curr_recall)
            prev_precision = curr_precision
            prev_recall = curr_recall
    if plot:
        plot_curve(precision_list,recall_list)
    return auc_score/10000.0
    
def compute_metrics(bboxes,gt_list):
    def get_key(item):
        return item[4]
    all_measures = zeros(0)
    all_truths = zeros(0,np.int8)
    gt_count = 0
    for bb_i, bb in enumerate(bboxes):
        gt = gt_list[bb_i]
        gt_count+=len(gt)
        if len(bb)>0:
            sorted_bb = array(sorted(bb,key=get_key,reverse=True))
            truths = zeros(bb.shape[0],np.int8)
            measures = sorted_bb[:,4]
            for gtrect in gt:
                for bbrect_i in range(sorted_bb.shape[0]):
                    if (truths[bbrect_i]==0) and \
                    isintersect(sorted_bb[bbrect_i,:4],gtrect):
                        truths[bbrect_i]=1
                        break
            all_measures = hstack((all_measures,measures))
            all_truths = hstack((all_truths,truths))
    if (all_measures.shape[0]==0):
        return 0.0
    
    #Call with plot=True to plot Precision-Recall curve
    return compute_auc(all_truths,all_measures,gt_count,plot=False)


if len(argv) != 3:
    stdout.write('Usage: %s train_dir test_dir\n' % argv[0])
    exit(1)

train_dir = argv[1]
test_dir = argv[2]

train_imgs, train_gt = load_gt(train_dir)
test_imgs, test_gt = load_gt(test_dir)

model = train_detector(train_imgs, train_gt)
bboxes = []
for img in test_imgs:
    bboxes.append(array(detect(model, img)))

stdout.write('%.2f\n' % compute_metrics(bboxes, test_gt))

