import numpy as np
import keras
from keras import optimizers
from keras.layers import DepthwiseConv2D
from keras.preprocessing import image
from keras_applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope

PATH_TEST_DATA = '/home/cardsmobile_data/_EAN_13/test'

'''
train_datagen= image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    directory=r"J:\Projects\CardsMobile\data\train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
'''
with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
    model = keras.models.load_model('../resource/models/my_model_base.h5')
# model = keras.models.load_model('my_model.h5')
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

test_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_generator = test_datagen.flow_from_directory(
    directory=PATH_TEST_DATA,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
test_generator.reset()
pred = model.predict_generator(test_generator, verbose=1)

filenames = test_generator.filenames
filenames = [file[file.find('/') + 1:][:-4] for file in filenames]  # cut .jpg
with open("../resource/data/filenames.txt", "w") as output:
    output.write(' '.join(filenames))


np.savetxt("../resource/data/predictions_on_test_data.csv", pred, delimiter=",")
