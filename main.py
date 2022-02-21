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

import graphviz
import pydot

from tensorflow.python.keras import models, layers


#path = "Resources/strain_resistors_type_1/norm_pos/IMG_7722.JPG"

# 5182, 3455
width_img, height_img = 1869, 1052
width_chip, height_chip = 400, 300


# default color scheme 47, 95, 52, 255, 73, 255
# 30, 94, 44, 201, 90, 154
min_hue, max_hue, min_sat, max_sat, min_val, max_val = 47, 95, 52, 255, 73, 255


def img2hsv(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
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
    # #cv2.imshow("1", img_gray)
    # #cv2.imshow("2", img_blur)
    # #cv2.imshow("3", img_canny)
    # #cv2.imshow("4", img_dilation)
    # #cv2.imshow("5", img_threshold)
    # ####cv2.waitKey(0)
    return img_threshold


def get_chip_contour(img): # ищет прямоугольник на выделенном с фото с помощью маски чипе
    # !!! point numeration - 0 -- 1
    #                        |    |
    #                        3 -- 2

    (y_chip, x_chip) = (390, 654)

    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_contour = cv2.imread("output/img_orig.jpg")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(img_contour, cnt, -1, (255,0,0), 1)
            cv2.fillPoly(img_contour, pts=[cnt], color=(255, 255, 255))  # Закрашивает контур платы
            peri = cv2.arcLength(cnt, True)
            approx_points = cv2.approxPolyDP(cnt, 0.05*peri, True)
            cv2.imwrite("output/chip_boundaries.jpg", img)

            type(approx_points)
            obj_corners = len(approx_points)
            x, y, w, h = cv2.boundingRect(approx_points)
            cv2.rectangle(img_contour, (x, y), (x+w, y+h), (0, 255, 2))
            rect = cv2.minAreaRect(cnt)
            chip_box = cv2.boxPoints(rect)
            chip_box_list = list(chip_box)

            # сортировка точек массива по порядку
            chip_box_list.sort(key=lambda x: x[0] * x[0] + x[1]*x[1])
            chip_box = np.array(chip_box_list)
            sum_corners = chip_box[0, 0] + chip_box[0, 1]

            # warping perspective of chip
            pts2 = np.float32([[0, 0], [0, y_chip], [x_chip, 0], [x_chip, y_chip]])
            matrix = cv2.getPerspectiveTransform(chip_box, pts2)

            img_orig = cv2.imread("output/img_orig.jpg")
            cropped_chip_jpg = cv2.warpPerspective(img_orig, matrix, (x_chip, y_chip))

            # картинка кропнутого чипа
            cv2.imwrite("output/chip_norm_pos.jpg", cropped_chip_jpg)
            #cv2.waitKey(0)


def get_resistor_images(chip_img, resistor_placement): # выделяет резисторы с обрезанного фото чипа
    (w_res, h_res) = (110, 40)

    for t in resistor_placement:
        print(t)
        t_num = int(t[0])
        res = chip_img[int(t[2]):int(t[2]) + h_res, int(t[1]):int(t[1]) + w_res]
        name = "output/res_"+str(t_num)+".jpg"
        cv2.imwrite(name, res)
        print(int(t[0]), name)


def resistor_detection(resistor_placement): # TODO должен определять сопротивление резисторова на каждом месте чипа
    name = ""
    (w_res, h_res) = (110, 40)
    for t in resistor_placement:
        t_num = int(t[0])
        chip_img = cv2.imread("output/chip_norm_pos.jpg")
        res = chip_img[int(t[2]):int(t[2]) + h_res, int(t[1]):int(t[1]) + w_res]
        name = "output/res_"+str(t_num)+".jpg"
        cv2.imwrite(name, res)


def process_photo(path):
    img = cv2.imread(path)
    img_orig = cv2.resize(img, (width_img, height_img))
    cv2.imwrite("output/img_orig.jpg", img_orig)
    img = cv2.resize(img, (width_img, height_img))

    masked_img = mask_img(img, min_hue, max_hue, min_sat, max_sat, min_val, max_val)
    cropped_chip_img = cv2.bitwise_and(img, img, mask=masked_img)
    chip_uncrop_masked = img_chip_select(cropped_chip_img)
    get_chip_contour(chip_uncrop_masked)
    chip_img = cv2.imread("output/chip_norm_pos.jpg")
    resistor_placement = genfromtxt('Resources/resistor_placement.csv', delimiter=';')

    get_resistor_images(chip_img, resistor_placement)


# min_hue, max_hue, min_sat, max_sat, min_val, max_val = img_color_calibration(img);

# (y_chip, x_chip) = (390, 654)  # chip dimensions
# (w_res, h_res) = (110, 40)  # resistor dimensions

photo_num = 1
path = "Resources/photo (" + str(photo_num) + ").jpg"

process_photo(path)

chip_img = cv2.imread("output/chip_norm_pos.jpg")
img_orig = cv2.imread("output/img_orig.jpg")

cv2.imshow("Detected chip", chip_img)
cv2.imshow("Original image", img_orig)

cv2.waitKey(0)

model = models.load_model('transistor_classifier.model')
