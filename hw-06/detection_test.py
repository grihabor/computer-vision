# -*- coding: utf-8 -*-

import numpy as np
import time
from sys import argv, stdout, exit
from skimage.io import imread
from detection import train_detector, detect
from PIL import Image, ImageDraw

FACEPOINTS_COUNT = 14


def load_data(path, return_paths=False):
    fi = open(path + '/gt.txt')
    lines = [line if line[-1] != '\n' else line[:-1]
             for line in fi.readlines()]
    fi.close()
    i = 0
    data = []
    gt = []
    if return_paths:
        impaths = []
    while i < len(lines) - FACEPOINTS_COUNT:
        imgdata = imread(path + '/images/' + lines[i], plugin='matplotlib')
        if return_paths:
            impaths.append(lines[i])
        if len(imgdata.shape) < 3:
            imgdata = np.array(
                [imgdata, imgdata, imgdata]).transpose((1, 2, 0))
        i += 1
        imggt = np.zeros((FACEPOINTS_COUNT, 2))
        for j in range(FACEPOINTS_COUNT):
            str_text = lines[i + j].split(';')
            nums = [int(s) for s in str_text]
            imggt[nums[0], :] = nums[1:]
        i += FACEPOINTS_COUNT
        data.append(imgdata)
        gt.append(imggt)
    gt = np.array(gt)
    if return_paths:
        return (data, gt, impaths)
    else:
        return (data, gt)


def compute_metrics(imgs, detected, gt):
    if len(detected) != len(gt):
        raise "Sizes don't match"
    diff = np.array(detected, dtype=np.float64) - np.array(gt)
    for i in range(len(imgs)):
        diff[i, :, 1] /= imgs[i].shape[0]
        diff[i, :, 0] /= imgs[i].shape[1]
    return np.sqrt(np.sum(diff ** 2) / (len(imgs) * 2 * FACEPOINTS_COUNT))


def visualise(imgs, detection_points, gt_points, res_dir, impaths,
              relative_radius=0.02,
              detection_color=(255, 0, 0),
              gt_color = (0, 255, 0)):
    for i in range(len(imgs)):
        pil_img = Image.fromarray(imgs[i])
        pil_draw = ImageDraw.Draw(pil_img)
        radius = relative_radius * min(pil_img.height, pil_img.width)
        for j in range(FACEPOINTS_COUNT):
            pt1 = detection_points[i, j, :]
            pt2 = gt_points[i, j, :]
            pil_draw.ellipse(
                (pt1[0] - radius, pt1[1] - radius, pt1[0] + radius, pt1[1] + radius), fill=detection_color)
            pil_draw.ellipse(
                (pt2[0] - radius, pt2[1] - radius, pt2[0] + radius, pt2[1] + radius), fill=gt_color)
        pil_img.save(res_dir + '/' + impaths[i])

if (len(argv) != 3) and (len(argv) != 5):
    stdout.write('Usage: %s train_dir test_dir [-v results_dir]\n' % argv[0])
    exit(1)
start_time = time.time()
train_dir = argv[1]
test_dir = argv[2]
visualisation_needed = (len(argv) > 3) and (argv[3] == '-v')
if visualisation_needed:
    res_dir = argv[4]
train_imgs, train_gt = load_data(train_dir)
model = train_detector(train_imgs, train_gt)
del train_imgs, train_gt
if visualisation_needed:
    test_imgs, test_gt, test_paths = load_data(test_dir, True)
else:
    test_imgs, test_gt = load_data(test_dir)
detection_results = np.array(detect(model, test_imgs))
print("Result: %.4f" % compute_metrics(test_imgs, detection_results, test_gt))
if visualisation_needed:
    visualise(test_imgs, detection_results, test_gt, res_dir, test_paths)
end_time = time.time()
print("Running time:", round(end_time - start_time, 2),
      's (' + str(round((end_time - start_time) / 60, 2)) + " minutes)")
