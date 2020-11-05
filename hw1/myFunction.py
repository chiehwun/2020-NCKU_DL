import cv2
import numpy as np


class TrackBar_1_4:
    def __init__(self, trackbar_name, win_title, slider_max):
        self.win_title = win_title
        self.slider_max = slider_max
        self.src1 = cv2.imread('Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
        self.src2 = cv2.flip(self.src1, flipCode=1)
        cv2.namedWindow(self.win_title)
        dst = cv2.addWeighted(self.src1, 0.0, self.src2, 1.0, 0.0)
        cv2.imshow(self.win_title, dst)
        cv2.createTrackbar(trackbar_name, self.win_title, 0,
                           self.slider_max, self.on_trackbar)

    def delfunc(self):
        cv2.destroyWindow(self.win_title)

    def on_trackbar(self, val):
        alpha = val / self.slider_max
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(self.src1, alpha, self.src2, beta, 0.0)
        cv2.imshow(self.win_title, dst)


# Useless
def convolution(img, ft, div, _min, _max):
    # img = np.array([[5, 5, 10, 10], [5, 5, 10, 10], [5, 5, 10, 10], [5, 5, 10, 10], [5, 5, 10, 10]])
    print('input_img:', img.shape)
    height = img.shape[0]
    width = img.shape[1]
    size = ft.shape[0]
    d = int((size - 1) / 2)
    img_res = np.zeros(img.shape, dtype=np.int)
    for y in range(height):
        for x in range(width):
            conv_sum = 0.0
            ft_sum = 0.0
            # print('y:', y, ', x:', x)
            for i in range(size):
                for j in range(size):
                    if 0 <= x - d + j < width and 0 <= y - d + i < height:
                        conv_sum += float(ft[i][j]) * \
                            float(img[y - d + i][x - d + j])
                        ft_sum += ft[i][j]
                        # print(j, i, ', conv_sum:', type(conv_sum), ', ft_sum:', ft_sum)
            if div is True:
                if ft_sum != 0:
                    img_res[y, x] = conv_sum / ft_sum
            else:
                img_res[y, x] = conv_sum

    # Normalize to 0-255 (uint8)
    print('img_res.max():', img_res.max(), ', img_res.min():',
          img_res.min(), np.average(img_res))
    # img_res = mapping(img_res, _min, _max)
    # print('img_out.max():', img_res.max(), ', img_out.min():', img_res.min(), np.average(img_res))
    img_res = img_res.astype(dtype=img.dtype)
    print('output_img:', img_res.shape)
    return img_res


# Useless
def mapping(arr, _min, _max):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) * (_max - _min) / (arr_max - arr_min) + _min
