#James Brodovsky
#from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os, cv2
import matplotlib.image as mpimg

# SOME CONSTANTS
IMG_H = 66
IMG_W = 200
IMG_C = 3
IMG_SHAPE = (IMG_H, IMG_W, IMG_C)

def build():
    # NVIDIA Model
    model = Sequential()
    model.add(Lambda(lambda x: x / 127 - 0.5, input_shape = IMG_SHAPE)) # Input layer and normalization
    model.add(Conv2D(24, (5,5), strides=(2,2), activation="elu"))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation="elu"))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation="elu"))
    model.add(Conv2D(64, (3,3), activation="elu"))
    model.add(Conv2D(64, (3,3), activation="elu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    #plot_model(model, to_file='modelVisual.png')
    return model

def train(model, rate, directory, xTrain, yTrain, xValid, yValid, batchSize, numEpochs):
    # Training pipeline
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = rate))
    ckpt = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    #model.fit(Xs, Ys, 32, epochs=Epochs, verbose=1, callbacks=[ckpt], validation_split=validSplit, shuffle=True)
    '''
    model.fit_generator(batch(directory, xTrain, yTrain, 128, True), 
                        32, numEpochs, max_q_size=1, 
                        validation_data=batch(directory, xValid, yValid, 128, False), 
                        nb_val_samples=len(xValid), callbacks=[ckpt], verbose=1)
                        '''
    samples = 1000 * batchSize
    model.fit_generator(batch(directory, xTrain, yTrain, batchSize, True), 
                        steps_per_epoch=np.ceil(samples/batchSize), 
                        epochs=numEpochs, 
                        verbose=1, 
                        callbacks=[ckpt], 
                        validation_data=batch(directory, xValid, yValid, batchSize, False), 
                        validation_steps=np.ceil(len(xValid)/batchSize))
                #, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    
def batch(directory, imagePaths, angles, size, isTraining):
    '''
    Python generator that yields the image path and associated steering angles.
    For training purposes, this data is augmented 50% of the time based off of raw captured data.
    '''
    images = np.empty([size, IMG_H, IMG_W, IMG_C])
    steer = np.empty(size)
    while True:
        i = 0
        for index in np.random.permutation(imagePaths.shape[0]):
            center, left, right = imagePaths[index]
            steeringAngle = angles[index]
            # if in a training session augment the image and angle
            image = []
            if isTraining and np.random.rand() < 0.5:
                image, steeringAngle = augment(directory, center, left, right, steeringAngle)
            else:
                image = preprocess(mpimg.imread(os.path.join(directory, center.strip())))
            # append image into to the batch, double checking to make sure it has been preprocessed
            try:
                images[i] = image
            except:
                image = preprocess(image)
                images[i] = image
            
            steer[i] = steeringAngle
            i += 1
            if i == size:
                break
        yield images, steer

def load(directory, validationSplit):
    #Load the raw data
    data = pd.read_csv(os.path.join(directory, 'driving_log.csv'))
    X = data[['center', 'left', 'right']].values
    Y = data['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=validationSplit, random_state=0)
    return X_train, X_valid, y_train, y_valid

def augment(directory, center, left, right, steering):
    # Augments the image for more robust training
    # Choose left, right, or center image and adjust steering angle interpretation
    choice = np.random.choice(3)
    image = np.zeros_like(center)
    angle = 0
    if choice == 0:
        image = mpimg.imread(os.path.join(directory, left.strip()))
        angle = steering + 0.2
    elif choice == 1:
        image = mpimg.imread(os.path.join(directory, right.strip()))
        angle = steering - 0.2
    else:
        image = mpimg.imread(os.path.join(directory, center.strip()))
        angle = steering
    
    # Randomly flip the image along the horizontal
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    
    # Randomly translate the image
    if np.random.rand() < 0.5:
        dx = 100*(np.random.rand() - 0.5)
        dy = 100*(np.random.rand() - 0.5)
        angle += dx*0.01
        M = np.float32([[1, 0, dx],[0,1,dy]])
        h,w = image.shape[:2]
        image = cv2.warpAffine(image, M, (w,h))
    
    return image, angle

def preprocess(image):
    # Preprocess an image to standardize the input to the machine learning
    image = image[60:-25, :,:] #crops the image to remove sky and front portion of the car
    image = cv2.resize(image, (IMG_W, IMG_H), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

if __name__ == '__main__':
    data_dir = 'data'
    epochs = 10
    batch_size = 32
    learningRate = 0.0001
    validSplit = 0.2
    
    print('Loading data')
    xT, xV, yT, yV= load(data_dir, validSplit)
    print('Building model')
    model = build()
    print('Beginning training')
    train(model, learningRate, data_dir, xT, yT, xV, yV, batch_size, epochs)   