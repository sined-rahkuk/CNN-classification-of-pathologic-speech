import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import pathlib
import split_folders

from keras.optimizers import SGD
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, AveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

PATH_TO_DATA = "wav_data"

# Python 3.6.8
# Keras	2.3.1
# Keras-Applications	1.0.8
# Keras-Preprocessing	1.1.0
# PyYAML	5.3.1
# SoundFile	0.10.3.post1
# audioread	2.1.8
# cffi	1.14.0
# cycler	0.10.0
# decorator	4.4.2
# h5py	2.10.0
# joblib	0.14.1
# kiwisolver	1.2.0
# librosa	0.7.2
# llvmlite	0.32.0
# matplotlib	3.2.1
# numba	0.48.0
# numpy	1.18.3
# pandas	1.0.3
# pip	18.1
# pycparser	2.20
# pyparsing	2.4.7
# python-dateutil	2.8.1
# pytz	2020.1
# resampy	0.2.2
# scikit-learn	0.22.2.post1
# scipy	1.4.1
# setuptools	40.6.2
# six	1.14.0
# split-folders	0.3.1
# TensorFlow 2.1.0

# Іспользуємо цю хуйню шоб конвертірувати аудіо файл у .png шоб потом його аналізірувати (converts WAV to png)
for g in ['healthy', 'pathological']:
    print('Creating png from wav')
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)  # Создасть папку img_data
    for filename in os.listdir(f'./{PATH_TO_DATA}/{g}'):
        if filename and filename.find('.png') != -1:
            continue
        songname = f'./{PATH_TO_DATA}/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=10)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default',
                     scale='dB')
        plt.axis('off')
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()

# 80 процетів хуйні для треніровки і 20 процентів хуйні для тестів (test(20) and train(80) data)
print('Split all data to train and test')
split_folders.ratio('./img_data/', output="./data", seed=1337, ratio=(.8, .2))

# Керасовська залупа яка рандомно мутірує пнгшку і получаються ліпші дата для треніровки
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # rescale all pixel values from 0-255, so after this step all our pixel values are in range (0,1)
    shear_range=0.2,  # to apply some random transformations
    zoom_range=0.2,  # to apply zoom
    horizontal_flip=True)  # image will be flipper horiz
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    './data/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False)
test_set = test_datagen.flow_from_directory(
    './data/val',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

# CNN модель, не знаю шо значать ці леєри но цю хуйню нам нада (CNN model)
model = Sequential()
input_shape = (64, 64, 3)
# 1st hidden layer
model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))
# 2nd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))
# 3rd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))
# Flatten
model.add(Flatten())
model.add(Dropout(rate=0.5))
# Add fully connected layer.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
# Output layer
model.add(Dense(1))
model.add(Activation('softmax'))
model.summary()

sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])

# Запускаємо модель (start model)
model.fit_generator(
    training_set,
    steps_per_epoch=100,
    epochs=2,
    validation_data=test_set,
    validation_steps=200)

# Це вже лиш результати (results)
model.evaluate_generator(generator=test_set, steps=50)

test_set.reset()
pred = model.predict_generator(test_set, steps=50, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

labels = training_set.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames = test_set.filenames
predictions = predictions[:len(filenames)]

results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv("prediction_results.csv", index=False)
