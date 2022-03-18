import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras import models, layers


from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

''' 
Разбитые по папкам фото резисторов из двух выборок - тестовая и тренировочная - подгружаются в модель, которая
затем обучается на фотографиях и сохраняется в проект.

В массив labels пишутся названия папок с разными типа резисторов
'''

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size_h, img_size_w)) # reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


def train_model():
    num_of_epochs = 400

    train = get_data('Resources/learning_resistors/training/')
    val = get_data('Resources/learning_resistors/testing/')

    l = []
    for i in train:
        print(i[1])
        for j in range(0, labels_len+1):
            if i[1] == j:
                l.append("type_"+str(j))

    # print(l)
    sns.set_style('darkgrid')
    sns.countplot(l)

    plt.figure(figsize=(5, 5))
    plt.imshow(train[1][0])
    plt.title(labels[train[0][1]])

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for feature, label in train:
      x_train.append(feature)
      y_train.append(label)



    for feature, label in val:
      x_val.append(feature)
      y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    x_train.reshape(-1, img_size_h, img_size_w, 1)
    y_train = np.array(y_train)

    x_val.reshape(-1, img_size_h, img_size_w, 1)
    y_val = np.array(y_val)


    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.3,  # Randomly zoom image
            width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

    datagen.fit(x_train)

    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_size_w, img_size_h, 3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(labels_len, activation="softmax"))  # 6 - number of categories

    model.summary()

    opt = Adam(lr=0.000001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=num_of_epochs, validation_data=(x_val, y_val))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(num_of_epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    #predictions = model.predict_classes(x_val)
    #predictions = predictions.reshape(1,-1)[0]
    #print(classification_report(y_val, predictions, target_names = ['No resistor (Class 0)', '1430 Om (Class 1)', '12 Om '
    #                                    '(Class 2)', '174000 Om (Class 3)', '1300 Om (Class 4)', '47200 kOm (Class 5)']))

    model.save('transistor_classifier.model')


labels = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5']
labels_len = len(labels)
img_size_h = 110
img_size_w = 40
train_model()
