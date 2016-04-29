import numpy as np

from sys import argv, stderr
from glob import iglob
from skimage.io import imread
from pickle import load

from seam_carve import seam_carve

def seam_coords(seam_mask):
    coords = np.where(seam_mask)
    t = [i for i in zip(coords[0], coords[1])]
    t.sort(key = lambda i: i[0])
    return tuple(t)

def msk_to_std(msk):
    return ((msk[:,:,0]!=0)*(-1) + (msk[:,:,1]!=0)).astype('int8')

if len(argv) != 3:
    stderr.write('Usage: %s mode input_dir\n' % argv[0])
    exit(1)

if argv[1] != '--base' and argv[1] != '--full':
    stderr.write('Usage: %s mode input_dir\n' % argv[0])
    exit(1)

mode = argv[1][2:]
input_dir = argv[2]

number = 0
TP = 0
for filename in iglob(input_dir + '/*.png'):
    if filename.find('_mask') >= 0:
        continue    

    print(filename)

    img = imread(filename)
    if mode == 'full':
        msk = imread(filename[:-4] + '_mask.png')
        msk = msk_to_std(msk)
        number += 8
    else:
        msk = None
        number += 2

    file = open(filename[:-4] + '_seams', 'rb')

    if mode == 'base':
        for orientation in ('horizontal', 'vertical'):
            TP += load(file) == seam_coords(seam_carve(img, orientation + ' shrink')[2])
    elif mode == 'full':
        for m in (None, msk):
            for direction in ('shrink', 'expand'):
                for orientation in ('horizontal', 'vertical'):
                    TP += load(file) == seam_coords(seam_carve(img, orientation + ' ' + direction, mask = m)[2])

    file.close()

print('Accuracy: {0:.2%}'.format(TP / number))

