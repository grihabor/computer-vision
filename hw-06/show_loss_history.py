from sys import argv
history = []
skip = 1
try:
    skip = int(argv[1])
    print('Using 1/{} of the data'.format(skip))
except:
    print('Using all the data')
with open('loss_history.txt', 'r') as ofile:
    for i, line in enumerate(ofile):
        if i % skip == 0:
            history.append(float(line.strip()))

import matplotlib.pyplot as plt
plt.plot(history)
plt.show()
