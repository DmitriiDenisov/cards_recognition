import os
import sys
import argparse

import keras
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing import image
from keras.layers.core import Activation, Reshape
from keras.layers import Conv2D
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
from engine.logger import TFLogger
from engine.tools.filesystem_functions import count_folders, get_barcode_class

""" Initialize argument parser """
parser = argparse.ArgumentParser(description='Model training script')
parser.add_argument('-pm', '--previous_model', action='store', type=str, default='',
                    help='Path to previously trained model to use. If empty than default model will be used.')
parser.add_argument('-bs', '--batch_size', action='store', type=int, default='',
                    help='Batch size to be used for training.')
parser.add_argument('-d', '--data_path', action='store', type=str, default='',
                    help='Path to folder with two subfolders: train and val. It is assumed that the folder has name of the barcode class (e.g. EAN_13, VOICE, CODE_128)')
parser.add_argument('-o', '--output_name', action='store', type=str, default='',
                    help='Name of the file where trained model will be stored. Full path ./models/$output_name$')
args = parser.parse_args()

""" Check if previously trained model is used """
if args.previous_model == '':
    TRAIN_FROM_ZERO = True
else:
    TRAIN_FROM_ZERO = False

""" Define data path and output path  """
DATA_PATH = args.data_path
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'models', args.output_name)
if os.path.exists(OUTPUT_PATH):
    raise "Model with name {} already exists.".format(args.output_name)

""" Enable logging for Tensorboard """
tf_logger = TFLogger(PROJECT_PATH, args.output_name, args.batch_size)
tf_logger.start()

""" Define barcode class and underlying classes number from file structure """
NUM_CLASSES = count_folders(os.path.join(args.data_path, 'train'))
BARCODE = get_barcode_class(args.data_path)

""" Modify existing architecture for actual number of classes """
if TRAIN_FROM_ZERO:
    # vgg_model = vgg16.VGG16(weights='imagenet')
    # inception_model = inception_v3.InceptionV3(weights='imagenet')
    # resnet_model = resnet50.ResNet50(weights='imagenet')
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    mobilenet_model.summary()
    mobilenet_model.layers.pop()
    mobilenet_model.layers.pop()
    mobilenet_model.layers.pop()
    mobilenet_model.get_layer(name='reshape_1').name = 'reshape_0'
    o = Conv2D(filters=NUM_CLASSES, kernel_size=(1, 1))(mobilenet_model.layers[-1].output)
    o = Activation('softmax', name='loss')(o)
    o = Reshape((NUM_CLASSES,))(o)

    mod_model = Model(mobilenet_model.input, o)
    mod_model.summary()
else:
    with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
        mod_model = keras.models.load_model(args.previous_model)

""" Data generators initialization: for train and validation sets """
train_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=False,
    zca_whitening=False)
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(args.data_path, 'train'),
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=args.batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=0,
    zca_whitening=False)

valid_generator = valid_datagen.flow_from_directory(
    directory=os.path.join(args.data_path, 'val'),
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=args.batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

""" Set train parameters for choosen model """
sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
mod_model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

""" Training """
checkpointer = ModelCheckpoint(OUTPUT_PATH, monitor='val_loss', verbose=0, save_best_only=False,
                               save_weights_only=False, mode='auto', period=1)
history = mod_model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=10,
                                  callbacks=[checkpointer, tf_logger.tbCallBack]
                                  )

""" Save logged entries """
tf_logger.save_local()
history.model.save(OUTPUT_PATH)
