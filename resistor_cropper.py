import cv2
import numpy as np
from numpy import genfromtxt

# 5182, 3455
width_img, height_img = 1869, 1052
width_chip, height_chip = 400, 300
res_type = 't_2'
resistor_placement = genfromtxt('Resources/' + res_type + '_resistor_coordinates.csv', delimiter=';')


# default color scheme 47, 95, 52, 255, 73, 255
# 30, 94, 44, 201, 90, 154
# min_hue, max_hue, min_sat, max_sat, min_val, max_val = 47, 95, 52, 255, 73, 255


def img2hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def empty(a):
    pass


# позволяет определить точный цвет искомой области (платы) с настройкой в реальном времени
def img_color_calibration(img):
    img = cv2.resize(img, (width_img, height_img))
    hsv_img = img2hsv(img)
    cv2.namedWindow("color options")
    cv2.resizeWindow("color options", 800, 500)
    cv2.createTrackbar("min hue", "color options", 0, 179, empty)  # 30
    cv2.createTrackbar("max hue", "color options", 179, 179, empty)  # 94
    cv2.createTrackbar("min sat", "color options", 0, 255, empty)  # 44
    cv2.createTrackbar("max sat", "color options", 255, 255, empty)  # 201
    cv2.createTrackbar("min val", "color options", 0, 255, empty)  # 90
    cv2.createTrackbar("max val", "color options", 255, 255, empty)  # 154

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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 5)
    img_canny = cv2.Canny(img_blur, 150, 80)
    kernel = np.ones((5, 5))
    img_dilation = cv2.dilate(img_canny, kernel, iterations=20)
    img_threshold = cv2.erode(img_dilation, kernel, iterations=3)

    return img_threshold


def get_chip_contour(img, img_orig):  # ищет прямоугольник на выделенном с фото с помощью маски чипе
    # !!! point numeration - 0 -- 1
    #                        |    |
    #                        3 -- 2

    (y_chip, x_chip) = (390, 654)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_orig_copy = img_orig.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(img_orig, cnt, -1, (255, 0, 0), 1)
            cv2.fillPoly(img_orig, pts=[cnt], color=(255, 255, 255))  # Закрашивает контур платы
            peri = cv2.arcLength(cnt, True)
            approx_points = cv2.approxPolyDP(cnt, 1 * peri, True)
            cv2.imwrite("output/chip_boundaries.jpg", img)

            type(approx_points)
            obj_corners = len(approx_points)
            x, y, w, h = cv2.boundingRect(approx_points)
            cv2.rectangle(img_orig, (x, y), (x + w, y + h), (0, 255, 2))
            rect = cv2.minAreaRect(cnt)
            chip_box = cv2.boxPoints(rect)
            chip_box_list = list(chip_box)

            # сортировка точек массива по порядку
            chip_box_list.sort(key=lambda x: x[0] * x[0] + x[1] * x[1])
            chip_box = np.array(chip_box_list)

            # warping perspective of chip
            pts2 = np.float32([[0, 0], [0, y_chip], [x_chip, 0], [x_chip, y_chip]])
            matrix = cv2.getPerspectiveTransform(chip_box, pts2)

            cropped_chip_jpg = cv2.warpPerspective(img_orig_copy, matrix, (x_chip, y_chip))

            # картинка кропнутого чипа
            cv2.imwrite("output/chip_norm_pos.jpg", cropped_chip_jpg)

            return (cropped_chip_jpg)


def get_resistor_images(chip_img, resistor_placement, path, j):  # выделяет резисторы с обрезанного фото чипа
    (w_res, h_res) = (110, 40)

    resistor_images = []
    for t in resistor_placement:
        t_num = int(t[0])
        res = chip_img[int(t[2]):int(t[2]) + h_res, int(t[1]):int(t[1]) + w_res]
        name = path + str(t_num) + "/res_" + str(j) + ".jpg"
        cv2.imwrite(name, res)
        resistor_images.append(res)


def crop_chip(img, img_path, path, j):
    img_orig = cv2.resize(img, (width_img, height_img))
    cv2.imwrite("output/img_orig.jpg", img_orig)
    img = cv2.resize(img, (width_img, height_img))

    #min_hue, max_hue, min_sat, max_sat, min_val, max_val = 70, 164, 111, 255, 82, 255 # 28.05.22
    min_hue, max_hue, min_sat, max_sat, min_val, max_val = 79,89,118,255,20,255 # t_1
    min_hue, max_hue, min_sat, max_sat, min_val, max_val = 82, 110, 124, 255, 15, 184  # t_2
    # min_hue, max_hue, min_sat, max_sat, min_val, max_val = 76,150,151,255,36,175
    # min_hue, max_hue, min_sat, max_sat, min_val, max_val = 47, 95, 52, 255, 73, 255
    masked_img = mask_img(img, min_hue, max_hue, min_sat, max_sat, min_val, max_val)

    uncropped_chip_img = cv2.bitwise_and(img, img, mask=masked_img)
    chip_uncrop_masked = img_chip_select(uncropped_chip_img)
    cropped_chip_img = get_chip_contour(chip_uncrop_masked, img_orig)
    cv2.imwrite(path + 'resistors/' + str(j) + '.jpg', cropped_chip_img)
    return cropped_chip_img

img3 = cv2.imread('C:/Users/wywycloud/PycharmProjects/course-resistor-position-control/Resources/sensor_videos/t_1/1/Frame0.jpg')
# img_color_calibration(img3)

sensor_num = 1
img_num = 1
def_path = 'C:/Users/wywycloud/PycharmProjects/course-resistor-position-control/Resources/sensor_videos/' + res_type + '/'
j = 0

while sensor_num < 17:
    #active_path = def_path + str(sensor_num) + '/resistors/'
    #img_path = active_path + 'chip (' + str(img_num) + ').jpg'
    active_path = def_path + str(sensor_num) + '/'
    img_path = active_path + 'Frame' + str(img_num - 1) + '.jpg'
    img = cv2.imread(img_path)
   # cv2.imshow('1',img)
  #  cv2.waitKey(0)
    img = crop_chip(img, img_path, active_path, j)

    if img is None:
        j = 0
        sensor_num += 1
        img_num = 1
        active_path = def_path + str(sensor_num) + '/resistors/'
        img_path = active_path + 'chip (' + str(img_num) + ').jpg'
        img = cv2.imread(img_path)
        img = crop_chip(img, img_path, active_path, j)

    get_resistor_images(img, resistor_placement, active_path, j)
    j += 1
    img_num += 1

