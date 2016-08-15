from sys import argv, exit

if len(argv) < 3:
    print('Usage:')
    print('python gtmake.py num_train num_test')
    exit()

indir = 'data/original'
out_train = 'train'
out_test = 'test'
filename = 'gt.txt'
num_train = int(argv[1])
num_test = int(argv[2])

import os

try:
    os.mkdir(out_train)
    os.mkdir(out_test)
except OSError:
    pass

out_train += '/' + filename
out_test += '/' + filename

with open(indir, 'r') as ifile:
    with open(out_train, 'w') as ofile:
        for line in ifile:
            if line.strip() == '{0:{fill}5}.jpg'.format(num_train, fill='0'):
                break
            ofile.write(line)
    with open(out_test, 'w') as ofile:
        if line.strip() == '{0:{fill}5}.jpg'.format(num_train+num_test, fill='0'):
            sys.exit()
        ofile.write(line)
        for line in ifile:
            if line.strip() == '{0:{fill}5}.jpg'.format(num_train+num_test, fill='0'):
                break
            ofile.write(line)
