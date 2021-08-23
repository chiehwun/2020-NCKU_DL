import os
import glob
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from skimage import io


# def check(filename):
#     try:
#       im = Image.load(filename)
#       im.verify()   # I perform also verify, don't know if he sees other types o defects
#       im.close()    # reload is necessary in my case
#       im = Image.load(filename)
#       im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#       im.close()
#     except:
#         print('ERR:', filename)


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        print(img_file)
        return False
    return True

if __name__ == '__main__':
    os.chdir('Datasets/PetImages')
    print(os.getcwd())
    # for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
    for filename in glob.glob('dog/*jpg'):
        print(filename)
        verify_image(filename)
    for filename in glob.glob('cat/*jpg'):
        verify_image(filename)

def split_data():
    os.chdir('Datasets/Q5_Image')
    print(os.getcwd())
    if os.path.isdir('train/dog') is False:
        os.makedirs('train/dog')
        os.makedirs('train/cat')
        os.makedirs('valid/dog')
        os.makedirs('valid/cat')
        os.makedirs('test/dog')
        os.makedirs('test/cat')
        tot_num = len(glob.glob('Cat/*.jpg'))
        train_num = int(tot_num * 0.6)
        valid_num = int(tot_num * 0.2)
        test_num = tot_num - train_num - valid_num
        print('total data: {}'.format(tot_num))
        print('pick {} as training dataset.'.format(train_num))
        print('pick {} as valid dataset.'.format(valid_num))
        print('pick {} as testing dataset.'.format(test_num))
        for c in random.sample(glob.glob('Cat/*.jpg'), train_num):
            shutil.move(c, 'train/cat')
        for c in random.sample(glob.glob('Cat/*.jpg'), valid_num):
            shutil.move(c, 'valid/cat')
        for c in random.sample(glob.glob('Cat/*.jpg'), test_num):
            shutil.move(c, 'test/cat')
        for c in random.sample(glob.glob('Dog/*.jpg'), train_num):
            shutil.move(c, 'train/dog')
        for c in random.sample(glob.glob('Dog/*.jpg'), valid_num):
            shutil.move(c, 'valid/dog')
        for c in random.sample(glob.glob('Dog/*.jpg'), test_num):
            shutil.move(c, 'test/dog')
        print('Splitting data complete.')
    os.chdir('../../')
    print(os.getcwd())
    # Create batches
    train_path = 'Datasets/Q5_Image/train'
    valid_path = 'Datasets/Q5_Image/valid'
    test_path = 'Datasets/Q5_Image/test'
    target_size = (250, 250)
    clss = ['cat', 'dog']
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)\
        .flow_from_directory(directory=train_path, target_size=target_size, classes=clss, batch_size=10)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)\
        .flow_from_directory(directory=valid_path, target_size=target_size, classes=clss, batch_size=10)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)\
        .flow_from_directory(directory=test_path, target_size=target_size, classes=clss, batch_size=10, shuffle=False)
    # Confirm batches status
    assert train_batches.n == 15000, 'error train_batches.n'
    assert valid_batches.n == 5000, 'error valid_batches.n'
    assert test_batches.n == 5000, 'error test_batches.n'
    imgs, labels = next(train_batches)

    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    plotImages(imgs)
    print(labels)
