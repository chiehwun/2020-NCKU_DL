import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt


def latest_timestamp():
    time_stamp = []
    for file in glob.glob("*.h5"):
        time_stamp.append(file[14:25])
    if len(time_stamp) == 0:
        print('Cannot find any model file!')
        exit(1)
    return time_stamp[-1]


if '__main__' == __name__:
    # files_dir = os.path.abspath(os.getcwd())
    model_filename = []
    hist_filename = []
    for file in glob.glob("*.h5"):
        model_filename.append(file)
    for file in glob.glob("*.json"):
        hist_filename.append(file)

    for name in hist_filename:
        with open(name, 'r') as file:
            j = json.load(file)
        fig = plt.figure(name)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(j['accuracy'])
        plt.plot(j['val_accuracy'])
        plt.legend(['Training', 'Testing'])
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy')
        plt.tight_layout()
        plt.subplot(2, 1, 2)
        plt.plot(j['loss'])
        plt.plot(j['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Loss')
        plt.legend(['Training', 'Testing'])
        plt.tight_layout()
        fig.show()
    plt.show()

        # plt.show()
        # val_acc = np.array(j['val_accuracy'])
        # trn_acc = np.array(j['accuracy'])
        # print(name+':')
        # print(np.round(val_acc[-5:-1], decimals=3))
        # print(np.round(trn_acc[-5:-1], decimals=3))
        # print()