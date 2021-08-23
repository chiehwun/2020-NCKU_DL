import myFunc
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QMessageBox, QLabel, QComboBox
from PyQt5 import uic
import platform
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import json
import keras.backend as backend         # show Hyperparameters
from tensorflow.keras.models import load_model
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor


class Hw2(QMainWindow):
    def __init__(self):
        super(Hw2, self).__init__()
        uic.loadUi("form.ui", self)
        # Q1.1
        self.bt_dc = self.findChild(QPushButton, 'bt_dc')
        self.lb_c1 = self.findChild(QLabel, 'lb_c1')
        self.lb_c2 = self.findChild(QLabel, 'lb_c2')
        self.bt_dc.clicked.connect(self.draw_ct)
        self.coinCount = []
        # Q1.2
        self.bt_cc = self.findChild(QPushButton, 'bt_cc')
        self.bt_cc.clicked.connect(self.count_coin)
        # Q2.1
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane
        self.bt_fc = self.findChild(QPushButton, 'bt_fc')
        self.bt_fc.clicked.connect(self.find_corn)
        # Q2.2
        self.bt_fi = self.findChild(QPushButton, 'bt_fi')
        self.bt_fi.clicked.connect(self.find_int)
        self.mtx = None     # camera matrix
        self.dist = None    # distortion coefficients
        self.rvecs = None   # rotation vectors
        self.tvecs = None   # translation vectors
        # Q2.3
        self.bt_fe = self.findChild(QPushButton, 'bt_fe')
        self.cb_sel = self.findChild(QComboBox, 'cb_sel')
        self.bt_fe.clicked.connect(self.find_ext)
        # Q2.4
        self.bt_fd = self.findChild(QPushButton, 'bt_fd')
        self.bt_fd.clicked.connect(self.find_dist)
        # Q3.1
        self.bt_ar = self.findChild(QPushButton, 'bt_ar')
        self.bt_ar.clicked.connect(self.ar)
        self.SlideShow = []
        # Q4.1
        self.bt_sdm = self.findChild(QPushButton, 'bt_sdm')
        self.bt_sdm.clicked.connect(self.stereo_disp_map)
        # Q5.1
        self.bt_5_1 = self.findChild(QPushButton, 'bt_5_1')
        self.bt_5_1.clicked.connect(self.Q5_1)
        self.model = None
        self.hist = None
        # Q5.2
        self.bt_5_2 = self.findChild(QPushButton, 'bt_5_2')
        self.bt_5_2.clicked.connect(self.Q5_2)
        # Q5.3
        self.bt_5_3 = self.findChild(QPushButton, 'bt_5_3')
        self.bt_5_3.clicked.connect(self.Q5_3)
        # Q5.4
        self.bt_5_4 = self.findChild(QPushButton, 'bt_5_4')
        self.bt_5_4.clicked.connect(self.Q5_4)
        self.test_batches = None
        self.model = None
        self.load_model()
        # Show Main Window
        self.show()

    def load_model(self):
        mod_list = glob.glob('model/*.h5')
        if len(mod_list) > 0:
            self.model = load_model(mod_list[-1])
            print('Successfully load \"{}\"'.format(mod_list[-1]))
        else:
            self.bt_5_1.setEnabled(False)
            self.bt_5_3.setEnabled(False)
            self.show_message('No model exists.')

        hist_list = glob.glob('model/*.json')
        if len(hist_list) > 0:
            with open(hist_list[-1]) as file:
                self.hist = json.load(file)
                print('Successfully load \"{}\".'.format((hist_list[-1])))
        else:
            self.bt_5_1.setEnabled(False)
            self.bt_5_3.setEnabled(False)
            self.show_message('No hist file exists.')

        self.hist = hist_list

    def draw_ct(self):     # Q1.1
        win_title = ['coin01.jpg', 'coin02.jpg']
        self.coinCount.clear()
        for coin in win_title:
            img_org = cv2.imread('Datasets/Q1_Image/{}'.format(coin))
            img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)    # Convert to gray
            # Gaussian filter
            img = cv2.GaussianBlur(img, (11, 11), 0)
            img = cv2.Canny(img, 120, 150)                    # Edge detection
            contours, hierarchy = cv2.findContours(
                img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.coinCount.append(len(contours))
            cv2.drawContours(img_org, contours, -1, (0, 0, 255), 2)
            cv2.imshow(coin, img_org)
        cv2.waitKey(0)
        for win in win_title:
            cv2.destroyWindow(win)

    def count_coin(self):   # Q1.2
        if len(self.coinCount) == 0:
            self.show_message('Please run \"1.1 Draw Contour\" first!')
            return
        self.lb_c1.setText(
            'There are {} coins in coin01.jpg'.format(self.coinCount[0]))
        self.lb_c2.setText(
            'There are {} coins in coin02.jpg'.format(self.coinCount[1]))

    def find_corn(self):    # Q2.1
        if len(self.objpoints) > 0:
            return
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane
        win_title = []
        for i in range(1, 16, 1):
            win_title.append('{}.bmp'.format(i))
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        # objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                  0:CHECKERBOARD[1]].T.reshape(-1, 2)
        for pic in win_title:
            img = cv2.imread('Datasets/Q2_Image/' + pic)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray @@
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            # If found, add object points, image points (after refining them)
            if ret:
                self.objpoints.append(objp)
                # refining pixel coordinates for given 2d points
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                # self.imgpoints.append(corners)
                self.imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(
                    img, CHECKERBOARD, corners, ret)
                print(pic)
                # cv2.namedWindow(pic, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                cv2.imshow(pic, img)
        cv2.waitKey(0)
        for pic in win_title:
            cv2.destroyWindow(pic)

    def find_int(self):     # Q2.2 @@
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            self.show_message('Please run \"2.1 Find Corners\' first.')
            return
        gray = cv2.imread('./Datasets/Q2_Image/1.bmp')
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)  # Convert to gray
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints,
                                                                               self.imgpoints,
                                                                               gray.shape[::-1], None, None)
        # print('RMS re-projection error:\n', ret)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, gray.shape[:2], 1, gray.shape[:2])
        float_formatter = "{:.6f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})
        print('intrinsic:\n', newcameramtx)
        # print('?????????:\n', self.mtx)

    def find_dist(self):    # Q2.3 @@
        if self.dist is None:
            self.show_message('Please run \"2.1 Find Corners\' first.')
            return
        print('distortion:\n', self.dist)

    def find_ext(self):     # Q2.4
        if self.dist is None:
            self.show_message('Please run \"2.1 Find Corners\' first.')
            return
        index = int(self.cb_sel.currentText())-1
        filename = '{}.bmp'.format(index+1)
        rve = cv2.Rodrigues(self.rvecs[index])  # transform to rotation matrix
        # print('rve:\n', rve)
        ext_mtx = np.concatenate((rve[0], self.tvecs[index]), axis=1)
        print('extrinsic matrix of \"{}\":\n'.format(filename), ext_mtx)

    def ar(self):           # Q3.1
        if len(self.SlideShow) > 0:
            self.slide_show()
            return
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane
        win_title = []
        for i in range(1, 6, 1):
            win_title.append('{}.bmp'.format(i))
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                  0:CHECKERBOARD[1]].T.reshape(-1, 2)
        corners2 = []
        img_org = []
        for pic in win_title:
            img_org.append(cv2.imread('Datasets/Q3_Image/' + pic))
            # Convert to gray
            gray = cv2.cvtColor(img_org[-1], cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points
                corners2.append(cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria))
                imgpoints.append(corners2[-1])
                print(pic)
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints,
                                                   imgpoints,
                                                   gray.shape[::-1], None, None)
        # Create pyramid vertex
        pyramid = np.float32(
            [[1, 1, 0], [3, 5, 0], [5, 1, 0], [3, 3, -3]]).reshape(-1, 3)
        for i in range(len(win_title)):
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2[i], mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(pyramid, rvecs, tvecs, mtx, dist)
            img = myFunc.draw(img_org[i], corners2, imgpts)
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
            self.SlideShow.append(img)
        self.slide_show()

    def slide_show(self):
        while True:
            for _slide in self.SlideShow:
                cv2.imshow('SlideShow', _slide)
                if cv2.waitKey(500) != -1:
                    cv2.destroyWindow('SlideShow')
                    return

    def stereo_disp_map(self):  # Q4.1 @@
        focal_length = 2826  # pixel
        baseline = 178  # mm
        dc = 123  # c_rx - c_lx
        imgL = cv2.imread('Datasets/Q4_Image/imgL.png', cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread('Datasets/Q4_Image/imgR.png', cv2.IMREAD_1RAYSCALE)
        stereo = cv2.StereoBM_create(numDisparities=160, blockSize=9)
        disp_img = stereo.compute(imgL, imgR)
        print('disp.dtype:', disp_img.dtype)
        print('disp.shape:', disp_img.shape)
        # Use matplotlib to implement user interaction
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Stereo Diaparity Map')
        title = plt.title('Disparity: -- pixels\nDepth: -- mm', loc='left')
        mark = plt.Circle((-10, -10), 10, color='r')
        ax.add_artist(mark)
        disp = disp_img.copy()
        ax.imshow(disp_img, 'gray')

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #       ('double' if event.dblclick else 'single', event.button,
            #        event.x, event.y, event.xdata, event.ydata))
            d = disp[int(event.ydata), int(event.xdata)]
            z = int(focal_length * baseline / (d + dc))
            title.set_text(
                'Disparity: {} pixels\nDepth: {} mm'.format(d + dc, z))
            mark.center = event.xdata, event.ydata
            fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        # cursor = Cursor(ax,
        #                 horizOn=True,
        #                 vrtOn=True,
        #                 color='green',
        #                 linewidth=2.0)

    def Q5_1(self):
        self.model.summary()
        img = cv2.imread('Datasets/Q5_Image/Q5_1.png')
        cv2.imshow('Q5.1', img)
        cv2.waitKey(0)
        cv2.destroyWindow('Q5.1')

    def Q5_2(self):
        win_title = ['Q5_2_1.png', 'Q5_2_2.png', 'Q5_2_3.png']
        for win in win_title:
            img = cv2.imread('Datasets/Q5_Image/' + win)
            cv2.imshow(win, img)
        cv2.waitKey(0)
        for win in win_title:
            cv2.destroyWindow(win)

    def Q5_3(self):
        clss = ['cat', 'dog']
        if self.test_batches is None:
            IMAGE_SIZE = (128, 128)
            test_datagen = ImageDataGenerator()
            self.test_batches = test_datagen.flow_from_directory(directory='Datasets/Q5_Image/test',
                                                                 target_size=IMAGE_SIZE,
                                                                 classes=clss,
                                                                 batch_size=1,
                                                                 shuffle=False)
        fig = plt.figure('Test Result', figsize=(15, 5), dpi=80)  # set window
        plt.clf()
        # plot image
        last_ind = len(self.test_batches.filenames) - 1
        test_ind = np.random.randint(0, last_ind)
        print('Test Image Index: {}/{}'.format(test_ind, last_ind))
        plt.subplot(1, 2, 1)
        print('filename:', self.test_batches.filenames[test_ind])
        # Use original image
        plt.imshow(plt.imread(self.test_batches.filepaths[test_ind]))
        plt.title(
            'Ans: ' + str(clss[np.argmax(self.test_batches[test_ind][1])]))
        plt.axis('off')
        # plot predict result bar chart
        plt.subplot(1, 2, 2)
        y_predict = self.model.predict(self.test_batches[test_ind])
        max_predict = np.argmax(y_predict)
        color = np.full(len(clss), 'C0')  # create ['C0', 'C0' ,...]
        color[max_predict] = 'C3'  # mark the largest as red
        plt.title('Predict Result: ' + clss[max_predict])
        plt.bar(clss, y_predict[0], color=color)
        fig.canvas.draw()
        plt.show()

    def Q5_4(self):
        fig = plt.figure('Q5.4', figsize=(15, 5), dpi=80)
        plt.title('Resize Augmentation Comparison')
        plt.bar(['Before Resize', 'After Resize'], [0.85, 0.91])
        plt.ylabel('Test dataset accuracy')
        plt.show()

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
    print("OpenCV version: {}".format(cv2.__version__))

    ######### TEST REGION #########

    ######### TEST REGION #########

    # Launch Qt window app
    app = QApplication([])
    widget = Hw2()
    sys.exit(app.exec_())
