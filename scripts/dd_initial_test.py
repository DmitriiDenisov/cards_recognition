from keras.applications import mobilenet
from keras.preprocessing import image
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers import Conv2D
from keras.models import Model
from keras import optimizers, callbacks
import os
import tensorflow as tf
from matplotlib import pyplot as plt

# python -m tensorboard.main --logdir ./
# cd ./Users/ddenisov/PycharmProjects/cardsmobile_recognition/logs/

tf.summary.FileWriterCache.clear()

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp_path_run = os.path.join(PROJECT_PATH, 'logs', 'run')
temp_path = temp_path_run + '1'
i = 2
while os.path.exists(temp_path):
    temp_path = temp_path_run + str(i)
    i += 1
log_run_dir = temp_path

NUM_CLASSES = 3

#sess = tf.Session()
#print(os.path.join(PROJECT_PATH, 'logs'))
#writer = tf.summary.FileWriter(os.path.join(PROJECT_PATH, 'logs'), sess.graph)
#sess = tf.InteractiveSession()
#writer = tf.summary.FileWriter(os.path.join(PROJECT_PATH, 'logs'))
#writer.add_graph(sess.graph)


print(log_run_dir)
#tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(PROJECT_PATH, 'logs'), histogram_freq=0,
#                            write_graph=True, write_images=True)

tbCallBack = callbacks.TensorBoard(log_dir=log_run_dir, histogram_freq=0, batch_size=32, write_graph=True,
                            write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None, embeddings_data=None)

sess = tf.InteractiveSession()
tf.summary.FileWriter(log_run_dir, sess.graph)

# vgg_model = vgg16.VGG16(weights='imagenet')
# inception_model = inception_v3.InceptionV3(weights='imagenet')
# resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

mobilenet_model.summary()
mobilenet_model.layers.pop()
mobilenet_model.layers.pop()
mobilenet_model.layers.pop()
mobilenet_model.get_layer(name='reshape_1').name='reshape_0'
o = Conv2D(filters=NUM_CLASSES, kernel_size=(1, 1))(mobilenet_model.layers[-1].output)
o = Activation('softmax', name='loss')(o)
o = Reshape((NUM_CLASSES,))(o)
# o.layers[-1].name = 'reshape_2'

mod_model = Model(mobilenet_model.input, o)
mod_model.summary()

train_datagen= image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
valid_datagen= image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    directory=r"../resource/samples/train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
valid_generator = valid_datagen.flow_from_directory(
    directory=r"../resource/samples/test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
mod_model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


STEP_SIZE_TRAIN = 2  # train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = 2  # valid_generator.n//valid_generator.batch_size
history = mod_model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    use_multiprocessing=False, callbacks=[tbCallBack]
)

saver = tf.train.Saver()
saver.save(sess, log_run_dir)

history.model.save('my_model.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('foo.png')