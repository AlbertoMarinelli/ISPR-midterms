from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import gradient_descent_v2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load train and test dataset (CIFAR-10)
def load_dataset():
    # load dataset (CIFAR-10)
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(data):
    # convert from integers to floats
    data_norm = data.astype('float32')
    # normalize to range 0-1
    data_norm = data_norm / 255.0
    return data_norm


# CNN model
def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    opt = gradient_descent_v2.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot learning curves (Loss, Accuracy)
def plot_learning_curves(history):
    # plot Loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot Accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    pyplot.savefig('learning_curves_plot.png')
    pyplot.close()


# Fit and test model
def run_test_model():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX = prep_pixels(trainX)
    testX = prep_pixels(testX)
    # define model
    model = define_model()
    # fit model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY))
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # plot learning curves
    plot_learning_curves(history)
    # save model
    model.save('final_model.h5')


# Test model
def test_model(path, testX, testY):
    # load model
    model = load_model(path)
    testX = prep_pixels(testX)
    # evaluate model on test dataset
    _, acc = model.evaluate(testX, testY)
    print('> %.3f' % (acc * 100.0))

'''
#Generate model
run_test_model()
'''
'''
#Test model
trainingX, trainingY, testX, testY = load_dataset()
test_model('final_model.h5', trainingX, trainingY)
'''