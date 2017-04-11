
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy
import os
from sys import path
from scipy.ndimage import imread
from scipy.misc import imresize


model_name = 'model'

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = path[0]+'/imgs/cats_and_dogs_medium/train'      # Path to training images
validation_data_dir = path[0]+'/imgs/cats_and_dogs_medium/test_util'  # Validation

nb_train_samples = 30000
nb_validation_samples = 900
batch_size = 16

# load saved model

# load json and create model
json_file = open(model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_name + ".h5")
print("Loaded model from disk")

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# Prepare data augmentation configuration

test_datagen = ImageDataGenerator(rescale=1. / 255)


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


output = model.predict_generator(validation_generator, steps=1)

if output[0][0] < 0.5:
    label = 'cat'
else:
    label = 'dog'

print("\nprediction value: {} -->\nclass: {}\n\n".format(str(output[0][0])[:6], label))