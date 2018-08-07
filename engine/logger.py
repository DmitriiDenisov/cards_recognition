import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard


class TFLogger:
    def __init__(self, project_path, batch_size):
        tf.summary.FileWriterCache.clear()
        self.project_path = project_path
        self.batch_size = batch_size
        self.session = tf.InteractiveSession()
        self.log_dir = self._create_run_folder()
        self.tbCallBack = self._init_callback()

    def _create_run_folder(self):
        if not os.path.exists(os.path.join(self.project_path, 'logs')):
            os.mkdir(os.path.join(self.project_path, 'logs'))

        temp_path_run = os.path.join(self.project_path, 'logs', 'run')
        temp_path = temp_path_run + '1'
        i = 2
        while os.path.exists(temp_path):
            temp_path = temp_path_run + str(i)
            i += 1
        return temp_path

    def _init_callback(self):
        return TensorBoard(log_dir=self.log_dir, histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                 embeddings_metadata=None, embeddings_data=None)

    def start(self):
        tf.summary.FileWriter(self.log_dir, self.session.graph)

    def save_local(self):
        self.saver = tf.train.Saver()
        self.saver.save(self.session, self.log_dir)

