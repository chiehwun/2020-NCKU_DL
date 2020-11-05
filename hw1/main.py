import sys
import cv2
import numpy as np
import myFunction as myFunc
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QMessageBox, QPlainTextEdit
from PyQt5 import uic


# noinspection SpellCheckingInspection
class Hw1(QMainWindow):
    def __init__(self):
        super(Hw1, self).__init__()
        uic.loadUi("form.ui", self)
        # Q1
        self.bt_load = self.findChild(QPushButton, "bt_load")
        self.bt_load.clicked.connect(self.load_img)
        self.bt_cs = self.findChild(QPushButton, "bt_cs")
        self.bt_cs.clicked.connect(self.color_sep)
        self.bt_img_flip = self.findChild(QPushButton, "bt_img_flip")
        self.bt_img_flip.clicked.connect(self.flip)
        self.bt_blend = self.findChild(QPushButton, "bt_blend")
        self.bt_blend.clicked.connect(self.blend)
        # Q2
        self.bt_mf = self.findChild(QPushButton, "bt_mf")
        self.bt_mf.clicked.connect(self.median_filter)
        self.bt_gb = self.findChild(QPushButton, "bt_gb")
        self.bt_gb.clicked.connect(self.gaussian_blur)
        self.bt_bf = self.findChild(QPushButton, "bt_bf")
        self.bt_bf.clicked.connect(self.bilateral_filter)
        # Q3
        self.bt_gb3 = self.findChild(QPushButton, "bt_gb3")
        self.bt_gb3.clicked.connect(self.gaussian_blur_3)
        self.bt_sx = self.findChild(QPushButton, "bt_sx")
        self.bt_sx.clicked.connect(self.sobel_x)
        self.bt_sy = self.findChild(QPushButton, "bt_sy")
        self.bt_sy.clicked.connect(self.sobel_y)
        self.bt_mag = self.findChild(QPushButton, "bt_mag")
        self.bt_mag.clicked.connect(self.mag)
        self.ft_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        self.ft_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
        self.ft_mag = np.sqrt(self.ft_x ** 2 + self.ft_y ** 2)
        self.gaussian_img = None
        self.sobel_x_img = None
        self.sobel_y_img = None
        self.mag_img = None
        # Q4
        self.bt_transform = self.findChild(QPushButton, "bt_transform")
        self.bt_transform.clicked.connect(self.transform)
        self.te_rot = self.findChild(QPlainTextEdit, "te_rot")
        self.te_scale = self.findChild(QPlainTextEdit, "te_scale")
        self.te_tx = self.findChild(QPlainTextEdit, "te_tx")
        self.te_ty = self.findChild(QPlainTextEdit, "te_ty")
        # Show Mainwindow
        self.show()

    def transform(self):
        win_title = ['Q4 Original', 'Q4 Image RST']
        img = cv2.imread('Dataset_opencvdl/Q4_Image/Parrot.png')
        cv2.imshow(win_title[0], img)
        # Get Parameters
        rotation = float(self.te_rot.toPlainText())
        scaling = float(self.te_scale.toPlainText())
        tx = int(self.te_tx.toPlainText())
        ty = int(self.te_ty.toPlainText())
        print(rotation, scaling, tx, ty)
        # Translating
        mat_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img_res = cv2.warpAffine(img, mat_trans, img.shape[:2])
        # Rotation & Scaling
        mat_rot = cv2.getRotationMatrix2D((160+tx, 84+ty), rotation, scaling)
        img_res = cv2.warpAffine(img_res, mat_rot, img_res.shape[:2])
        cv2.imshow(win_title[1], img_res)
        cv2.waitKey(0)
        for win in win_title:
            cv2.destroyWindow(win)

    def mag(self):  # Q3.4
        if self.gaussian_img is None:
            self.show_message('Please run \"3.1 Gaussian Blur\" first!')
            return
        if self.sobel_x_img is None:
            self.show_message('Please run \"3.2 Sobel X\" first!')
            return
        if self.sobel_y_img is None:
            self.show_message('Please run \"3.3 Sobel Y\" first!')
            return
        win_title = 'Q3.3 Magnitude'
        self.mag_img = np.sqrt(self.sobel_x_img.astype(dtype=np.int) ** 2 + self.sobel_y_img.astype(dtype=np.int) ** 2)
        self.mag_img = self.mag_img.astype(dtype=self.gaussian_img.dtype)
        cv2.imshow(win_title, self.mag_img)
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)

    def sobel_y(self):  # Q3.3
        if self.gaussian_img is None:
            self.show_message('Please run \"3.1 Gaussian Blur\" first!')
            return
        win_title = 'Q3.2 Sobel Y'
        self.sobel_y_img = cv2.filter2D(self.gaussian_img, -1, self.ft_y)
        cv2.imshow(win_title, self.sobel_y_img)
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)
        return

    def sobel_x(self):  # Q3.2
        if self.gaussian_img is None:
            self.show_message('Please run \"3.1 Gaussian Blur\" first!')
            return
        win_title = 'Q3.2 Sobel X'
        self.sobel_x_img = cv2.filter2D(self.gaussian_img, -1, self.ft_x)
        cv2.imshow(win_title, self.sobel_x_img)
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)

    def gaussian_blur_3(self):  # Q3.1
        win_title = ['Q3.1 Original', 'Q3.1 Gaussian Blur']
        img = cv2.imread('Dataset_opencvdl/Q3_Image/Chihiro.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imshow(win_title[0], img)
        # Create filter
        size = 3
        sigma = 0.707
        ft = np.zeros((size, size))
        d = int((size - 1) / 2)
        for i in range(0, size):
            for j in range(0, size):
                ft[i][j] = -((j - d) ** 2 + (i - d) ** 2)
        ft = np.exp(ft) / (2 * np.pi * sigma * sigma)
        self.gaussian_img = cv2.filter2D(img, -1, ft)
        cv2.imshow(win_title[1], self.gaussian_img)
        cv2.waitKey(0)
        for win in win_title:
            cv2.destroyWindow(win)

    @staticmethod
    def median_filter():    # Q2.1
        win_title = ['Q2.1 Original', 'Q2.1 Median Blur']
        img = cv2.imread('Dataset_opencvdl/Q2_Image/Cat.png')
        cv2.imshow(win_title[0], img)
        img_mb = cv2.medianBlur(img, 7)
        cv2.imshow(win_title[1], img_mb)
        cv2.waitKey(0)
        for win in win_title:
            cv2.destroyWindow(win)

    @staticmethod
    def gaussian_blur():    # Q2.2
        win_title = ['Q2.2 Original', 'Q2.2 Gaussian Blur']
        img = cv2.imread('Dataset_opencvdl/Q2_Image/Cat.png')
        cv2.imshow(win_title[0], img)
        img_gb = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imshow(win_title[1], img_gb)
        cv2.waitKey()
        for win in win_title:
            cv2.destroyWindow(win)

    @staticmethod
    def bilateral_filter():     # Q2.3
        win_title = ['Q2.3 Original', 'Q2.3 Bilateral Blur']
        img = cv2.imread('Dataset_opencvdl/Q2_Image/Cat.png')
        cv2.imshow(win_title[0], img)
        img_bf = cv2.bilateralFilter(img, 9, 90, 90)
        cv2.imshow(win_title[1], img_bf)
        cv2.waitKey()
        for win in win_title:
            cv2.destroyWindow(win)

    @staticmethod
    def load_img():
        win_title = 'Q1.1'
        img = cv2.imread('Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
        cv2.imshow(win_title, img)
        print("Height = ", str(img.shape[0]))
        print("Width = ", str(img.shape[1]))
        cv2.waitKey()
        cv2.destroyWindow(win_title)

    @staticmethod
    def color_sep():
        img = cv2.imread('Dataset_opencvdl/Q1_Image/Flower.jpg')
        win_title = ['Blue', 'Green', 'Red', 'Q1.2']
        cv2.imshow(win_title[-1], img)
        for i in range(0, 3):
            img_new = np.zeros(img.shape, dtype=img.dtype)
            img_new[:, :, i] = img[:, :, i]
            cv2.imshow(win_title[i], img_new)
        print('Press any key to close the windows...')
        cv2.waitKey(0)
        for win in win_title:
            cv2.destroyWindow(win)

    @staticmethod
    def flip():
        win_title = ['Q1.3 Original', 'Q1.3 Result']
        img = cv2.imread('Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
        cv2.imshow(win_title[0], img)
        img_flip = cv2.flip(img, flipCode=1)
        cv2.imshow(win_title[1], img_flip)
        print('Press any key to close the windows...')
        cv2.waitKey()
        for win in win_title:
            cv2.destroyWindow(win)

    @staticmethod
    def blend():
        trackbar = myFunc.TrackBar_1_4('BLEND', 'Q1.4 Blend', 255)
        cv2.waitKey(0)
        trackbar.__del__()

    @staticmethod
    def show_message(text):
        msg = QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setText(text)
        msg.setIcon(QMessageBox.Warning)
        x = msg.exec_()


if __name__ == "__main__":
    app = QApplication([])
    widget = Hw1()
    sys.exit(app.exec_())
