import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from shutil import copyfile
import random
import zipfile

try:
    shutil.rmtree('C:\\Data\\CV\\dogs-vs-cats\\')
except:
    print("Directory Does not exists")

# This code block unzips the full Cats-v-Dogs dataset to /tmp
# which will create a tmp/PetImages directory containing subdirectories
# called 'Cat' and 'Dog' (that's how the original researchers structured it)
path_cats_and_dogs = 'C:\\Data\\CV\\dogs-vs-cats.zip'

local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:\\Data\\CV\\dogs-vs-cats\\')
zip_ref.close()

test_path_cats_and_dogs = 'C:\\Data\\CV\\dogs-vs-cats\\test1.zip'

local_zip = test_path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:\\Data\\CV\\dogs-vs-cats\\')
zip_ref.close()

train_path_cats_and_dogs = 'C:\\Data\\CV\\dogs-vs-cats\\train.zip'

local_zip = train_path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:\\Data\\CV\\dogs-vs-cats\\')
zip_ref.close()

BASE_PATH = 'C:\\Data\\CV\\dogs-vs-cats\\train\\'
TRAIN_PATH = 'C:\\Data\\CV\\dogs-vs-cats\\train_data\\'
VAL_PATH = 'C:\\Data\\CV\\dogs-vs-cats\\validation_data\\'

try:
    shutil.rmtree(TRAIN_PATH)
    shutil.rmtree(VAL_PATH)
except:
    print("Directory does not exists")

Directory
does
not exists

os.mkdir(TRAIN_PATH)
os.mkdir(VAL_PATH)
train_dog = os.path.join(TRAIN_PATH, 'dog')
train_cat = os.path.join(TRAIN_PATH, 'cat')
val_dog = os.path.join(VAL_PATH, 'dog')
val_cat = os.path.join(VAL_PATH, 'cat')
print(train_cat)
os.mkdir(train_dog)
os.mkdir(train_cat)
os.mkdir(val_dog)
os.mkdir(val_cat)

C:\Data\CV\dogs - vs - cats\train_data\cat

# segregating cats and dog files for training and validation
cat_file_list = []
dog_file_list = []
for file in os.listdir(BASE_PATH):
    if file[:3] == 'cat':
        cat_file_list.append(file)
    elif file[:3] == 'dog':
        dog_file_list.append(file)
# shuffling the list

train_split = 0.8

random.sample(cat_file_list, len(cat_file_list))
random.sample(dog_file_list, len(dog_file_list))
num_of_training_files_cat = int(len(cat_file_list) * train_split)
num_of_training_files_dog = int(len(dog_file_list) * train_split)
print("Total training cat files", num_of_training_files_cat)
print("Training files dog", num_of_training_files_dog)
print("Total filess", len(cat_file_list) + len(dog_file_list))
print("Total val Dog files", len(dog_file_list) - num_of_training_files_dog)

for file in cat_file_list[:num_of_training_files_cat]:
    copyfile(os.path.join(BASE_PATH, file), os.path.join(train_cat, file))

for file in cat_file_list[num_of_training_files_cat:]:
    copyfile(os.path.join(BASE_PATH, file), os.path.join(val_cat, file))

for file in dog_file_list[:num_of_training_files_dog]:
    copyfile(os.path.join(BASE_PATH, file), os.path.join(train_dog, file))

for file in dog_file_list[num_of_training_files_dog:]:
    copyfile(os.path.join(BASE_PATH, file), os.path.join(val_dog, file))
print("Total val Dog files", len(dog_file_list) - num_of_training_files_dog)

batch_size = 64  # 128
epochs = 75
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.3

)  # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=TRAIN_PATH,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=VAL_PATH,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)



sample_training_images, _ = next(train_data_gen)

plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



model.summary()

batch_size=100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=1000,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=1000
)