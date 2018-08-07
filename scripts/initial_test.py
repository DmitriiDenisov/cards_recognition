import os

import tensorflow as tf
import keras
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing import image
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers import Conv2D
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope

TRAIN_FROM_ZERO = False

tf.summary.FileWriterCache.clear()

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp_path_run = os.path.join(PROJECT_PATH, 'logs', 'run')
temp_path = temp_path_run + '1'
i = 2
while os.path.exists(temp_path):
    temp_path = temp_path_run + str(i)
    i += 1
log_run_dir = temp_path

tbCallBack = TensorBoard(log_dir=log_run_dir, histogram_freq=0, batch_size=32, write_graph=True,
                            write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None, embeddings_data=None)

sess = tf.InteractiveSession()
tf.summary.FileWriter(log_run_dir, sess.graph)


NUM_CLASSES = 2663

# vgg_model = vgg16.VGG16(weights='imagenet')
# inception_model = inception_v3.InceptionV3(weights='imagenet')
# resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

mobilenet_model.summary()
mobilenet_model.layers.pop()
mobilenet_model.layers.pop()
mobilenet_model.layers.pop()
mobilenet_model.get_layer(name='reshape_1').name='reshape_0'
o = Conv2D(filters=NUM_CLASSES, kernel_size=(1,1))(mobilenet_model.layers[-1].output)
o = Activation('softmax', name='loss')(o)
o = Reshape((NUM_CLASSES,))(o)
# o.layers[-1].name = 'reshape_2'

mod_model = Model(mobilenet_model.input, o)
mod_model.summary()

train_datagen= image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=False,
        zca_whitening=False)
valid_datagen= image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=0,
        zca_whitening=False)

train_generator = train_datagen.flow_from_directory(
    directory=r"../resource/train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
valid_generator = valid_datagen.flow_from_directory(
    directory=r"../resource/test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


if not TRAIN_FROM_ZERO:
    with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
        mod_model = keras.models.load_model('my_model_batch8.h5')

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
mod_model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

checkpointer = ModelCheckpoint('my_model_batch8_zca.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


history = mod_model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    callbacks=[checkpointer, tbCallBack]
)


saver = tf.train.Saver()
saver.save(sess, log_run_dir)

history.model.save('my_model_batch8_zca.h5')