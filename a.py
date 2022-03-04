import cv2
import numpy as np
from keras.utils.vis_utils import plot_model
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# import graphviz
# import pydot

from tensorflow.python.keras import models, layers

# 5182, 3455
width_img, height_img = 1869, 1052
width_chip, height_chip = 400, 300


# default color scheme 47, 95, 52, 255, 73, 255
# 30, 94, 44, 201, 90, 154
# min_hue, max_hue, min_sat, max_sat, min_val, max_val = 47, 95, 52, 255, 73, 255


def img2hsv(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imwrite("output/chip_hsv.jpg", img)
    return img


def empty(a):
    pass


def img_color_calibration(img):  #позволяет определить точный цвет искомой области (платы) с настройкой в реальном времени
    img = cv2.resize(img, (width_img, height_img))
    hsv_img = img2hsv(img)
    cv2.namedWindow("color options")
    cv2.resizeWindow("color options", 800, 500)
    cv2.createTrackbar("min hue", "color options", 0, 179, empty)       #30
    cv2.createTrackbar("max hue", "color options", 179, 179, empty)     #94
    cv2.createTrackbar("min sat", "color options", 0, 255, empty)       #44
    cv2.createTrackbar("max sat", "color options", 255, 255, empty)     #201
    cv2.createTrackbar("min val", "color options", 0, 255, empty)       #90
    cv2.createTrackbar("max val", "color options", 255, 255, empty)     #154

    while True:
        min_hue = cv2.getTrackbarPos("min hue", "color options")
        max_hue = cv2.getTrackbarPos("max hue", "color options")
        min_sat = cv2.getTrackbarPos("min sat", "color options")
        max_sat = cv2.getTrackbarPos("max sat", "color options")
        min_val = cv2.getTrackbarPos("min val", "color options")
        max_val = cv2.getTrackbarPos("max val", "color options")
        print(min_hue, max_hue, min_sat, max_sat, min_val, max_val)
        lower = np.array([min_hue, min_sat, min_val])
        upper = np.array([max_hue, max_sat, max_val])

        mask = cv2.inRange(hsv_img, lower, upper)

        cv2.imshow("hsv", hsv_img)
        cv2.imshow("original", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)

    return min_hue, max_hue, min_sat, max_sat, min_val, max_val


def mask_img(img, min_hue, max_hue, min_sat, max_sat, min_val, max_val):
    lower = np.array([min_hue, min_sat, min_val])
    upper = np.array([max_hue, max_sat, max_val])
    hsv_img = img2hsv(img)
    mask = cv2.inRange(hsv_img, lower, upper)
    return mask


def img_chip_select(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 100, 100)
    kernel = np.ones((5, 5))
    img_dilation = cv2.dilate(img_canny, kernel,iterations=2)
    img_threshold = cv2.erode(img_dilation, kernel,iterations=1)
    return img_threshold


def get_chip_contour(img, img_orig): # ищет прямоугольник на выделенном с фото с помощью маски чипе
    # !!! point numeration - 0 -- 1
    #                        |    |
    #                        3 -- 2

    (y_chip, x_chip) = (390, 654)

    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_orig_copy = img_orig.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(img_orig, cnt, -1, (255,0,0), 1)
            cv2.fillPoly(img_orig, pts=[cnt], color=(255, 255, 255))  # Закрашивает контур платы
            peri = cv2.arcLength(cnt, True)
            approx_points = cv2.approxPolyDP(cnt, 0.05*peri, True)
            cv2.imwrite("output/chip_boundaries.jpg", img)

            type(approx_points)
            obj_corners = len(approx_points)
            rect = cv2.minAreaRect(cnt)
            chip_box = cv2.boxPoints(rect)
            chip_box_list = list(chip_box)

            # сортировка точек массива по порядку
            chip_box_list.sort(key=lambda x: x[0] * x[0] + x[1]*x[1])
            chip_box = np.array(chip_box_list)

            # warping perspective of chip
            pts2 = np.float32([[0, 0], [0, y_chip], [x_chip, 0], [x_chip, y_chip]])
            matrix = cv2.getPerspectiveTransform(chip_box, pts2)

            cropped_chip_jpg = cv2.warpPerspective(img_orig_copy, matrix, (x_chip, y_chip))

            # картинка кропнутого чипа
            cv2.imwrite("output/chip_norm_pos.jpg", cropped_chip_jpg)

            return(cropped_chip_jpg)


def get_resistor_images(chip_img, resistor_placement):  # выделяет резисторы с обрезанного фото чипа
    (w_res, h_res) = (110, 40)

    resistor_images = []
    for t in resistor_placement:
        t_num = int(t[0])
        res = chip_img[int(t[2]):int(t[2]) + h_res, int(t[1]):int(t[1]) + w_res]
        name = "output/res_"+str(t_num)+".jpg"
        cv2.imwrite(name, res)
        resistor_images.append(res)

    return resistor_images


model = models.load_model('resistor-position-control/transistor_classifier.model')


def process_photo(img):
    img_orig = cv2.resize(img, (width_img, height_img))
    cv2.imwrite("output/img_orig.jpg", img_orig)
    img = cv2.resize(img, (width_img, height_img))

    min_hue, max_hue, min_sat, max_sat, min_val, max_val = 47, 95, 52, 255, 73, 255
    masked_img = mask_img(img, min_hue, max_hue, min_sat, max_sat, min_val, max_val)

    uncropped_chip_img = cv2.bitwise_and(img, img, mask=masked_img)
    chip_uncrop_masked = img_chip_select(uncropped_chip_img)
    cropped_chip_img = get_chip_contour(chip_uncrop_masked, img_orig)

    resistor_placement = genfromtxt('resistor-position-control/Resources/resistor_placement.csv', delimiter=';')

    resistor_images = []
    resistor_images = get_resistor_images(cropped_chip_img, resistor_placement)

    labels = {0: "Резистор отсутствует",
              1: "R = 1430 Ом",
              2: 'R = 12 Ом',
              3: 'R = 174000 Ом',
              4: 'R = 1300 Ом',
              5: 'R = 47200 Ом'}

    results = {}
    count = 0
    for r in resistor_images:
        # img = np.array(r) / 255
        img = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        prediction = model.predict(np.array([img]) / 255,batch_size=1)
        results[count] = ([labels[np.argmax(prediction)], prediction[0][np.argmax(prediction)]])
        count += 1


    return cropped_chip_img, resistor_images, results


# img_read = cv2.imread("Resources/photo (1).jpg")
# cv2.imshow("resistor", img_read)
# cv2.waitKey(500)
#
# img_processed, resistor_images = process_photo(img_read)
#
# for res in resistor_images:
#     cv2.imshow("resistor", res)
#     cv2.waitKey(500)