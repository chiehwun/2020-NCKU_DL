import sys
import platform
import glob
import os
import keras
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QMessageBox, QSpinBox
from PyQt5 import uic
import json                             # read training history file
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10      # load data sets
import keras.backend as backend         # show Hyperparameters
from tensorflow.keras.models import load_model
import tensorflow as tf


def latest_timestamp():
    time_stamp = []
    for file in glob.glob("*.h5"):
        time_stamp.append(file[14:25])
    if len(time_stamp) == 0:
        print('Cannot find any model file!')
        exit(1)
    return time_stamp[-1]


class Hw1_5(QMainWindow):
    def __init__(self):
        super(Hw1_5, self).__init__()
        uic.loadUi("form.ui", self)
        time_stamp = latest_timestamp()
        print('time_stamp:', time_stamp)
        self.model_filename = 'vgg16_cifar10_'+time_stamp+'.h5'
        self.hist_filename = 'vgg16_cifar10_history_'+time_stamp+'.json'
        # load data sets
        self.x_train_org = None
        self.x_train = None
        self.y_train = None
        self.x_test_org = None
        self.x_test = None
        self.y_test = None
        self.load_cifar10()
        self.label = ['plane', 'car', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        # load trained model
        self.model = None
        self.load_trained_model()
        self.hist = None
        self.load_hist()
        # Q5.1
        self.bt_img = self.findChild(QPushButton, 'bt_img')
        self.bt_img.clicked.connect(self.load_img)
        # Q5.2
        self.bt_hyp = self.findChild(QPushButton, 'bt_hyp')
        self.bt_hyp.clicked.connect(self.show_hyp_par)
        # Q5.3
        self.bt_mod_str = self.findChild(QPushButton, 'bt_mod_str')
        self.bt_mod_str.clicked.connect(self.show_mod_str)
        # Q5.4
        self.bt_acc = self.findChild(QPushButton, 'bt_acc')
        self.bt_acc.clicked.connect(self.show_acc)
        # Q5.5
        self.sb_index = self.findChild(QSpinBox, 'sb_index')
        self.bt_test = self.findChild(QPushButton, 'bt_test')
        self.bt_test.clicked.connect(self.test)
        self.bt_randTest = self.findChild(QPushButton, 'bt_randTest')
        self.bt_randTest.clicked.connect(self.randTest)
        # Show Main Window
        self.show()

    def load_cifar10(self):
        print('Loading cifar10......')
        (self.x_train_org, self.y_train), (self.x_test_org,
                                           self.y_test) = cifar10.load_data()
        self.x_train = self.x_train_org.astype('float32')
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test_org.astype('float32')
        self.x_test = self.x_test / 255.0
        print('x_train_org.shape:', self.x_train_org.shape, self.x_train_org.dtype)
        print('x_train.shape:', self.x_train.shape, self.x_train.dtype)
        print('y_train.shape:', self.y_train.shape, self.y_train.dtype)
        print('x_test_org.shape:', self.x_test_org.shape, self.x_test_org.dtype)
        print('x_test.shape:', self.x_test.shape, self.x_test.dtype)
        print('y_test.shape:', self.y_test.shape, self.y_test.dtype)
        print('Complete')

    def load_trained_model(self):
        if os.path.isfile(self.model_filename):
            self.model = load_model(self.model_filename)
        else:
            self.show_message('\'' + self.model_filename + '\' cannot find.')

    def load_hist(self):
        if os.path.isfile(self.hist_filename):
            with open(self.hist_filename) as file:
                self.hist = json.load(file)
        else:
            self.show_message('File: \''+self.hist_filename+'\' cannot found.')

    def load_img(self):     # Q5.1
        rand_num = np.random.randint(0, self.x_train_org.shape[0] - 1, 10)

        fig = plt.figure('Show Train Images')
        plt.clf()
        for i in range(0, rand_num.shape[0]):
            plt.subplot(2, 5, i+1)
            plt.imshow(self.x_train_org[rand_num[i]])
            plt.title(self.label[self.y_train[rand_num[i]][0]])
            plt.axis('off')
        fig.canvas.draw()
        plt.show()

    def show_hyp_par(self):  # Q5.2
        if self.model is None:
            self.show_message('Model not found!')
            return
        if self.hist is None:
            self.show_message('History not found!')
            return
        print('\nhyperparameters:')
        print('batch size:', int(
            np.ceil(self.x_train.shape[0]/self.hist['steps'])))
        print('learning rate:', backend.eval(self.model.optimizer.lr))
        print('optimizer:', self.model.optimizer.get_config()['name'])

    def show_mod_str(self):  # Q5.3
        if self.model is None:
            self.show_message('Model not found!')
            return
        print('')
        print(self.model.summary())

    def show_acc(self):     # Q5.4
        if self.hist is None:
            self.show_message('History not found!')
            return
        fig = plt.figure('Accuracy')
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(self.hist['accuracy'])
        plt.plot(self.hist['val_accuracy'])
        plt.legend(['Training', 'Testing'])
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy')
        plt.tight_layout()
        plt.subplot(2, 1, 2)
        plt.plot(self.hist['loss'])
        plt.plot(self.hist['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Loss')
        plt.legend(['Training', 'Testing'])
        plt.tight_layout()
        fig.canvas.draw()
        plt.show()

    def test(self):         # Q5.5
        test_ind = self.sb_index.value()
        ####### Test Region #######
        # y_predict = self.model.predict(np.array([self.x_test[test_ind]]))
        # print(y_predict[0])
        ####### Test Region #######

        fig = plt.figure('Test Result', figsize=(15, 5), dpi=80)  # set window
        plt.clf()

        # plot image
        plt.subplot(1, 2, 1)
        plt.imshow(self.x_test_org[test_ind])
        plt.axis('off')
        print('Test Image Index:', test_ind)
        plt.title('Ans: ' + self.label[self.y_test[test_ind][0]])

        # plot predict result bar chart
        plt.subplot(1, 2, 2)
        # Remind "predict()" might report some error
        # y_predict = np.random.random(10) # For testing
        y_predict = self.model.predict(np.array([self.x_test[test_ind]]))
        max_predict = np.argmax(y_predict[0])
        title = 'Predict Result: ' + self.label[max_predict]
        plt.title(title)
        color = np.full(len(self.label), 'C0')
        color[max_predict] = 'C3'
        plt.bar(self.label, y_predict[0], color=color)
        fig.canvas.draw()
        plt.show()

    def randTest(self):
        test_ind = np.random.randint(0, self.x_test_org.shape[0] - 1)
        self.sb_index.setValue(test_ind)
        self.test()

    @staticmethod
    def show_message(text):
        msg = QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setText(text)
        msg.setIcon(QMessageBox.Warning)
        x = msg.exec_()


if __name__ == "__main__":
    # Tensorflow configuration
    print("Platform: {}".format(platform.platform()))
    print("Tensorflow version: {}".format(tf.__version__))
    print("Keras version: {}".format(keras.__version__))

    # # Test without Jupyter notebook running (Success)
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # time_stamp = '11-09_11_19'
    # new_model = load_model('vgg16_cifar10_' + time_stamp + '.h5')
    # new_model.summary()
    # pred_res = new_model.predict(np.array([x_test[0]]))
    # for i in pred_res:
    #     print(i)

    # # Solution
    # config = tf.compat.v1.ConfigProto(gpu_options=
    #                                   tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95)
    #                                   # device_count = {'GPU': 1}
    #                                   )
    # config.gpu_options.allow_growth = False
    # session = tf.compat.v1.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(session)

    # Launch Qt window app
    app = QApplication([])
    widget = Hw1_5()
    sys.exit(app.exec_())

# ======== Unused function ========
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
