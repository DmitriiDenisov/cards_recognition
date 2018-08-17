import os
import tensorflow as tf
from keras.callbacks import TensorBoard

class TFLogger(TensorBoard):
    def __init__(self, project_path, model_name, batch_size, log_every=1, **kwargs):
        tf.summary.FileWriterCache.clear()
        self.project_path = project_path
        self.model_name = model_name[:-3]
        self.batch_size = batch_size
        self.session = tf.InteractiveSession()
        self.log_dir = self._create_run_folder()

        super().__init__(log_dir=self.log_dir, batch_size=self.batch_size, **kwargs)
        self.log_every = log_every
        self.counter = 0
        self.sum_loss = 0
        self.sum_acc = 0
        self.sum_pre = 0
        self.counter_for_mean = 1
        self.epoch_end = False

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.epoch_end:
            self.epoch_end = False
            self.counter_for_mean = 1
            self.counter_for_mean_acc = 1
            self.sum_loss = 0
            self.sum_acc = 0
            self.sum_pre = 0
        self.sum_loss += logs['loss']
        self.sum_acc += logs['acc']
        self.sum_pre += logs['precision']
        mean_loss = self.sum_loss / self.counter_for_mean
        mean_acc = self.sum_acc / self.counter_for_mean
        self.counter_for_mean += 1

        if self.counter % self.log_every == 0:
            logs['mean_loss'] = mean_loss
            logs['mean_acc'] = mean_acc
            logs['train_on_batch_loss'] = logs.pop('loss')
            logs['train_on_batch_acc'] = logs.pop('acc')
            logs['train_on_batch_pre'] = logs.pop('precision')
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end = True
        for name, value in logs.items():
            if (name in ['batch', 'size']) or ('val' not in name):
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def _create_run_folder(self):
        #PATH_TO_LOGS_ON_SERVER = 'C:\\Users\ddenisov\PycharmProjects\cardsmobile_recognition\logs'
        PATH_TO_LOGS_ON_SERVER = os.path.join(os.path.dirname(os.path.dirname(self.project_path)), 'cardsmobile_data', 'logs')
        if not os.path.exists(PATH_TO_LOGS_ON_SERVER):
            os.mkdir(PATH_TO_LOGS_ON_SERVER)

        temp_path_run = os.path.join(PATH_TO_LOGS_ON_SERVER, 'run')
        temp_path = temp_path_run + '1' + '_' + self.model_name
        i = 2
        while os.path.exists(temp_path):
            temp_path = temp_path_run + str(i) + '_' + self.model_name
            i += 1
        return temp_path

    def start(self):
        tf.summary.FileWriter(self.log_dir, self.session.graph)

    #def save_local(self):
    #    self.saver = tf.train.Saver()
    #    self.saver.save(self.session, self.log_dir)

