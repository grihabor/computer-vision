# -*- coding: utf-8 -*-

import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import keras

FACEPOINTS_COUNT = 14
INPUT_SIZE = 112
INPUT_SHAPE = (1, INPUT_SIZE, INPUT_SIZE)


FINAL_VERSION = True
batch_size = 100
nb_epoch = 200


if not FINAL_VERSION:
    colors = [  
                #eyebrows
                [0.5, 0.,  0.],
                [1.,  0.,  0.],
                [1.,  0.5, 0.],
                [1.,  1.,  0.],
                #left eye
                [0.5, 1.,  0.],
                [0.,  1.,  0.],
                [0.,  1.,  0.5],
                #right eye
                [0.,  1.,  1.],
                [0.,  0.5, 1.],
                [0.,  0.,  0.5],
                #nose
                [0.,  0.,  1.],
                #mouth
                [0.5, 0.,  1.],
                [1.,  0.,  1.],
                [1.,  0.,  0.5]  ]

    import matplotlib.pyplot as plt
    def show_image_with_facial_points(img, gt, subplot):
        img_1 = gray2rgb(img)
        for (x, y), color in zip(gt * INPUT_SIZE, colors):
            img_1[y, x] = color
        plt.subplot(subplot)
        plt.imshow(img_1, cmap='gray', interpolation='none')
        

def get_model():
    model = Sequential()
    # 112 * 112 * 1
    model.add(Convolution2D(
        nb_filter=32, 
        nb_row=3, 
        nb_col=3, 
        border_mode='valid', 
        input_shape=INPUT_SHAPE
    ))
    # 110 * 110 * 32
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 55 * 55 * 32
    model.add(Convolution2D(
        nb_filter=64, 
        nb_row=2, 
        nb_col=2, 
        border_mode='valid'
    ))
    # 54 * 54 * 64
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 27 * 27 * 64    
    model.add(Convolution2D(
        nb_filter=128, 
        nb_row=2, 
        nb_col=2, 
        border_mode='valid'
    ))
    # 26 * 26 * 128
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 13 * 13 * 128
    model.add(Flatten())
    model.add(Dense(600))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(600))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * FACEPOINTS_COUNT))
    model.add(Activation('tanh'))
    
    model.compile(
        loss='mse',
        optimizer='adadelta',
        metrics=['accuracy']
    )
    return model

def train_detector(imgs, gts):

    model = get_model()
    #model.save_weights('trained_model.h5')
    
    '''
    model.load_weights('final_model.hdf5')
    return model    
    '''

    data_norm = [[  [[x / img.shape[1], y / img.shape[0]] for x, y in gt],
                rgb2gray(resize(img, (INPUT_SIZE, INPUT_SIZE))),
                img.shape[0]]
                    for gt, img in zip(gts, imgs)]
    
    gts_original = [data[0] for data in data_norm]
    gts_mirrored = [[
                        #eyebrows
                        [1. - gt[3][0], gt[3][1]],
                        [1. - gt[2][0], gt[2][1]],
                        [1. - gt[1][0], gt[1][1]],
                        [1. - gt[0][0], gt[0][1]],
                        #left eye
                        [1. - gt[9][0], gt[9][1]],
                        [1. - gt[8][0], gt[8][1]],
                        [1. - gt[7][0], gt[7][1]],
                        #right eye
                        [1. - gt[6][0], gt[6][1]],
                        [1. - gt[5][0], gt[5][1]],
                        [1. - gt[4][0], gt[4][1]],
                        #nose
                        [1. - gt[10][0], gt[10][1]],
                        #mouth                        
                        [1. - gt[13][0], gt[13][1]],
                        [1. - gt[12][0], gt[12][1]],
                        [1. - gt[11][0], gt[11][1]]   
                            ] for gt in gts_original]

    imgs_original = [data[1] for data in data_norm]
    imgs_mirrored = [np.fliplr(img) for img in imgs_original]

    imgs_norm = np.array(imgs_original + imgs_mirrored)
    gts_norm = np.array(gts_original + gts_mirrored)
     
    '''
    for i in range(len(imgs_original)):

        show_image_with_facial_points(imgs_norm[0 + i], 
                                        gts_norm[0 + i], 
                                        121)
        show_image_with_facial_points(imgs_norm[len(imgs_original) + i], 
                                        gts_norm[len(imgs_original) + i], 
                                        122)
        plt.show()
    '''
    

    del data_norm

    class LossHistory(keras.callbacks.Callback):
        
        def __init__(self):
            with open('loss_history.txt', 'w'):
                pass
        
        def on_train_begin(self, logs={}):  
            self.losses = []

        def save_to_file(self):
            if self.losses == []:
                return
            with open('loss_history.txt', 'a') as ofile:
                for line in self.losses:
                    ofile.write('{}\n'.format(line))
            self.losses = []
    
        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            if len(self.losses) > 100:
                self.save_to_file()
          

    print('images: {}'.format(imgs_norm.shape))

    X_train = imgs_norm.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))
    Y_train = gts_norm.reshape((-1, 2 * FACEPOINTS_COUNT))

    callbacks = []
    if not FINAL_VERSION:
        history = LossHistory()
        checkpointer = keras.callbacks.ModelCheckpoint(filepath="./weights.hdf5", verbose=0, save_best_only=True)
        callbacks.append(history)
        callbacks.append(checkpointer)

    if FINAL_VERSION:
        validation_split = 0.
        verbose = 0
    else:
        validation_split = 0.1
        verbose = 2
    model.fit(
        X_train, 
        Y_train, 
        batch_size = batch_size, 
        nb_epoch = nb_epoch, 
        verbose = verbose, 
        validation_split = validation_split,
        callbacks = callbacks,
        shuffle = True
    )

    if not FINAL_VERSION:
        history.save_to_file()

    print('model is ready')
    if not FINAL_VERSION:
        model.save_weights('trained_model.h5', overwrite = True)
   
    return model


def detect(model, imgs, gts=None):
    
    data_norm = [[img.shape, rgb2gray(resize(img, (INPUT_SIZE, INPUT_SIZE)))] for img in imgs]

    imgs_norm = np.array([data[1] for data in data_norm])
    img_shapes = np.array([data[0] for data in data_norm])


    X_test = imgs_norm.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))
    prediction = model.predict(X_test, verbose=0)
    prediction = prediction.reshape((-1, FACEPOINTS_COUNT, 2))

    prediction = np.array([[[x * img_shape[1], y * img_shape[0]] 
                                for x, y in gt] 
                                    for gt, img_shape in zip(prediction, img_shapes)])
    
    if not FINAL_VERSION:
        import matplotlib.pyplot as plt
        for i in range(100):
            plt.subplot(121)
            plt.title('Real data')
            original = gray2rgb(imgs_norm[i])    
            t = np.array([[x / img_shapes[i][1], y / img_shapes[i][0]] for x, y in gts[i]]) * INPUT_SIZE
            #print(t)
            for x, y in t:
                original[y, x] = [1., 0., 0.]
            plt.imshow(original, cmap='gray', interpolation='None')
            
            plt.subplot(122)
            plt.title('Predicted')
            result = gray2rgb(imgs_norm[i])
            for x, y in np.array([[x / img_shapes[i][1], y / img_shapes[i][0]] for x, y in prediction[i]]) * INPUT_SIZE:
                result[y, x] = [1., 0., 0.]
            plt.imshow(result, cmap='gray', interpolation='None')
            
            plt.show()
            plt.savefig('saved.png')

    
    return prediction




