import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.python.keras import models, layers


from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

''' 
Folded photos of resistors from the samples - testing and training - are loaded into the model, which
then learns from photographs and is being saved into the project.

The label arrays contain the names of folders with particles like resistors. 
'''
tf.device('gpu')


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


def myprint(s):
    with open('modelsummary' + str(u) + '.txt', 'a') as f:
        print(s, file=f)


def train_model(conv_1_num, conv_2_num, conv_3_num, conv_4_num, dense_num, fourth_conv_layer,):
    num_of_epochs = 84

    train = get_data('Resources/learning/training/')
    val = get_data('Resources/learning/testing/')

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
    plt.imshow(train[4][4])
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

    model.add(Conv2D(conv_1_num, 3, padding="same", activation="relu", input_shape=(img_size_w, img_size_h, 3)))
    model.add(MaxPool2D())

    model.add(Conv2D(conv_2_num, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(conv_3_num, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    if fourth_conv_layer:
        model.add(Conv2D(conv_4_num, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(dense_num, activation="relu"))
    model.add(Dense(labels_len, activation="softmax"))  # 6 - number of categories

    opt = Adam(lr=0.000001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=num_of_epochs, validation_data=(x_val, y_val))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(num_of_epochs)

    plt.rcdefaults()
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'Times New Roman'
    # Tell matplotlib to use the locale we set above
    plt.rcParams['axes.formatter.use_locale'] = True

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, color="black", label='Точность в обучающей выборкеь')
    plt.plot(epochs_range, val_acc, color="blue", label='Точность в валидационной выборке')
    plt.legend(loc='lower right')
    plt.title('Точности в обучающей и валидационной выборках')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, color="black", label='Потери в обучающей выборке')
    plt.plot(epochs_range, val_loss, color="blue", label='Потери в валидационной выборке')
    plt.legend(loc='upper right')
    plt.title('Потери в обучающей и валидационной выборках')
    plt.grid()
    plt.savefig('Neural_network_' + str(u) + '.png')
    plt.show()
    plt.draw()

    model.summary(print_fn=myprint)

    model.save('resistor_classifier.model')

def visualize_filters(model):
    # Iterate thru all the layers of the model
    for layer in model.layers:
        if 'conv' in layer.name:
            weights, bias = layer.get_weights()
            #print(layer.name, filters.shape)

            # normalize filter values between  0 and 1 for visualization
            f_min, f_max = weights.min(), weights.max()
            filters = (weights - f_min) / (f_max - f_min)
            print(filters.shape[3])
            filter_cnt = 1

            # plotting all the filters
            for i in range(filters.shape[3]):
                # get the filters
                filt = filters[:, :, :, i]
                # plotting each of the channel, color image RGB channels
                for j in range(filters.shape[0]):
                    ax = plt.subplot(filters.shape[3], filters.shape[0], filter_cnt)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(filt[:, :, j])
                    filter_cnt += 1
            plt.show()


def visualise_feature_maps():
    img_path = '\\dogs-vs-cats\\test1\\137.jpg'  # dog
    # Define a new Model, Input= image
    # Output= intermediate representations for all layers in the
    # previous model after the first.
    successive_outputs = [layer.output for layer in
                          model.layers[1:]]  # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)  # Load the input image
    img = load_img(img_path, target_size=(150, 150))  # Convert ht image to Array of dimension (150,150,3)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)  # Rescale by 1/255
    x /= 255.0  # Let's run input image through our vislauization network
    # to obtain all intermediate representations for the image.
    successive_feature_maps = visualization_model.predict(
        x)  # Retrieve are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        print(feature_map.shape)
        if len(feature_map.shape) == 4:

            # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers

            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # Postprocess the feature to be visually palatable
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # Tile each filter into a horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x  # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
labels = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7', 'type_8', 'type_9', 'type_10',
          'type_11', 'type_12', 'type_13', 'type_14']
labels_len = len(labels)
img_size_h = 110
img_size_w = 40

u = 10
train_model(16, 32, 64, 0, 64, False)

fifth_layer = False
'''
u = 1
train_model(32, 32, 64, 0, 64, False)

u += 1
train_model(64, 32, 16, 0, 32, False)

u += 1
train_model(32, 16, 16, 16, 64, True)

u += 1
train_model(128, 64, 32, 16, 64, True)

u += 1
train_model(128, 32, 8, 32, 64, True)

u += 1
train_model(16, 32, 64, 0, 64, False)

u += 1
train_model(32, 32, 64, 0, 128, False)

u += 1
train_model(64, 32, 64, 0, 64, False)

u += 1
train_model(64, 128, 32, 0, 64, False)

#model = models.load_model('resistor_classifier.model')
#visualize_filters(model)
#model.summary()
#visualise_feature_maps()
'''
