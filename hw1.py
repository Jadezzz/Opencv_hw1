# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow

import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget

import numpy as np


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        # load image
        img = cv2.imread('images/dog.bmp')

        # show image, print image size
        cv2.imshow('Image', img)
        print('Height =', img.shape[0])
        print('Width =', img.shape[1])

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_2_click(self):
        # load image
        img = cv2.imread('images/color.png')

        # show original image
        cv2.imshow('Original', img)

        # color conversion
        b, g, r = cv2.split(img)
        new_img = cv2.merge((g, r, b))

        # show new image
        cv2.imshow('New Image', new_img)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_3_click(self):
        # load image
        img = cv2.imread('images/dog.bmp')

        # flip image
        flipped_img = cv2.flip(img, 1)

        # display image
        cv2.imshow('Original', img)
        cv2.imshow('Flipped', flipped_img)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn1_4_click(self):
        # load image1 and image2
        img1 = cv2.imread('images/dog.bmp')
        img2 = cv2.flip(img1, 1)

        # create callback func for trackerbar
        def blend(x):
            # get blending rate
            percentage = cv2.getTrackbarPos('BLEND', 'BLENDING')
            alpha = 1 - percentage / 100
            beta = percentage / 100

            # compute blended image
            img_blended = cv2.addWeighted(img1, alpha, img2, beta, 0)
            cv2.imshow('BLENDING', img_blended)

        # create window and trackbar
        cv2.namedWindow('BLENDING')
        cv2.createTrackbar('BLEND', 'BLENDING', 0, 100, blend)
        cv2.imshow('BLENDING', img1)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def on_btn2_1_click(self):
        # load image and convert it to greyscale
        img = cv2.imread('images/M8.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3x3 Gaussian smooth
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # show blurred image
        cv2.imshow('Gaussian Blur', img)

        # Sobel Edge detection
        # gX, gY are 3x3 Sobel kernal
        gX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        gY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # output dimension
        row = img.shape[0] - 2
        col = img.shape[1] - 2

        # output img
        vert_img = np.zeros((row, col), dtype=np.int32)
        hori_img = np.zeros((row, col), dtype=np.int32)
        magnitude = np.zeros((row, col), dtype=np.int32)
        angle = np.zeros((row, col))

        # convolution
        for i in range(row):
            for j in range(col):
                crop = img[i:i+3, j:j+3]
                vert_img[i, j] = np.sum(np.multiply(crop, gX))
                hori_img[i, j] = np.sum(np.multiply(crop, gY))
                magnitude[i, j] = np.sqrt(np.square(vert_img[i, j]) + np.square(hori_img[i, j]))
                angle[i, j] = np.arctan2(hori_img[i, j], vert_img[i, j])

        # normalize
        vert_img_abs = cv2.convertScaleAbs(vert_img)
        hori_img_abs = cv2.convertScaleAbs(hori_img)
        magnitude_abs = cv2.convertScaleAbs(magnitude)

        # map angle from +- pi/2 to 0~pi
        angle = np.rad2deg(angle)
        angle = (angle < 0)*360 + angle

        # threshold
        threshold = 40

        vert_img_out = (vert_img_abs > threshold) * vert_img_abs
        hori_img_out = (hori_img_abs > threshold) * hori_img_abs
        magnitude_out = (magnitude_abs > threshold) * magnitude_abs

        # horizontal and vertical edges
        cv2.imshow('Vertical Edges', vert_img_out)
        cv2.imshow('Horizontal Edges', hori_img_out)

        # create callback function for trackerbar
        def changeMagnitude(x):
            # get threshold
            th = cv2.getTrackbarPos('Threshold', 'Magnitude')

            # thresholding
            out = (magnitude_abs > th) * magnitude_abs
            cv2.imshow('Magnitude', out)

        # create magnitude window
        cv2.namedWindow('Magnitude')
        cv2.createTrackbar('Threshold', 'Magnitude', 40, 255, changeMagnitude)
        cv2.imshow('Magnitude', magnitude_out)

        # select image
        def selectImage(angle_deg):
            # calculate the range of angle
            angle_min = angle_deg - 10
            angle_max = angle_deg + 10

            # handle situation which angle < 0 or angle > 360
            if angle_min < 0:
                angle_min = angle_min + 360
                image_out = (angle > angle_min) * magnitude_out + (angle < angle_max) * magnitude_out
            elif angle_max > 360:
                angle_max = angle_max - 360
                image_out = (angle > angle_min) * magnitude_out + (angle < angle_max) * magnitude_out
            else:
                image_out = (angle > angle_min) * (angle < angle_max) * magnitude_out

            image_out = cv2.convertScaleAbs(image_out)

            return image_out

        # create callback function for trackerbar
        def changeDirection(x):
            # get direction
            angle_deg = cv2.getTrackbarPos('Angle', 'Direction')

            # select range
            image_out = selectImage(angle_deg)

            # show image
            cv2.imshow('Direction', image_out)

        # create direction window
        cv2.namedWindow('Direction')
        cv2.createTrackbar('Angle', 'Direction', 10, 360, changeDirection)

        # call show the first time with default value 10
        cv2.imshow('Direction', selectImage(10))

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn3_1_click(self):
        # load image
        Gau_0 = cv2.imread('images/pyramids_Gray.jpg')

        # compute level 1 and level 2 Gaussian Pyramids
        Gau_1 = cv2.pyrDown(Gau_0)
        Gau_2 = cv2.pyrDown(Gau_1)

        # compute level 0 and level 1 Laplacian Pyramids
        Lap_0 = cv2.subtract(Gau_0, cv2.pyrUp(Gau_1))
        Lap_1 = cv2.subtract(Gau_1, cv2.pyrUp(Gau_2))

        # conpute level 0 and level 1 inverse Pyramids
        Inv_1 = cv2.add(Lap_1, cv2.pyrUp(Gau_2))
        Inv_0 = cv2.add(Lap_0, cv2.pyrUp(Inv_1))

        # show images
        cv2.imshow('1) Gaussian Pyramid level 1', Gau_1)
        cv2.imshow('2) Laplacian Pyramid level 0', Lap_0)
        cv2.imshow('3) Inverse Pyramid level 1', Inv_1)
        cv2.imshow('4) Inverse Pyramid level 0', Inv_0)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn4_1_click(self):
        # load image
        img = cv2.imread('images/QR.png')

        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # show Original Image
        cv2.imshow('Original Image', img)

        # define global threshold func
        def threshold(img):
            ret, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
            return thresh

        # threshold image
        img_out = threshold(img)

        # show image
        cv2.imshow('Thresholded Image', img_out)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn4_2_click(self):
        # load image
        img = cv2.imread('images/QR.png')

        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # show Original Image
        cv2.imshow('Original Image', img)

        # define adaptive threshold function
        def adaptiveThreshold(img):
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -1)
            return thresh

        # threshold image
        img_out = adaptiveThreshold(img)

        # show image
        cv2.imshow('Thresholded Image', img_out)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn5_1_click(self):
        # load image
        img = cv2.imread('images/OriginalTransform.png')
        cv2.imshow('Original Image', img)

        # get arguments
        Tx = self.edtTx.text()
        Ty = self.edtTy.text()
        angle = self.edtAngle.text()
        scale = self.edtScale.text()

        # check '' value
        if Tx == '' or Ty == '' or angle == '' or Ty == '':
            print('Please enter a value for each box!!!!')
            return

        # parse type
        Tx = int(Tx)
        Ty = int(Ty)
        angle = int(angle)
        scale = float(scale)

        rows, cols, channels = img.shape

        # Translation
        M1 = np.float32([[1, 0, Tx], [0, 1, Ty]])
        out = cv2.warpAffine(img, M1, (cols, rows))

        center = (130 + Tx, 125 + Ty)

        # Rotation and Scaling
        M2 = cv2.getRotationMatrix2D(center, angle, scale)
        out = cv2.warpAffine(out, M2, (cols, rows))

        # output result
        cv2.imshow('Rotation + Scale + Translation', out)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_btn5_2_click(self):
        # load image
        img = cv2.imread('images/OriginalPerspective.png')

        # take a copy of image
        img_clear = img.copy()

        global count
        count = 0
        pts1 = np.empty((4, 2), np.float32)
        pts2 = np.float32([[20, 20], [450, 20], [450, 450], [20, 450]])

        # define mouse callback function
        def record_points(event, x, y, flags, param):
            global count
            if event == cv2.EVENT_LBUTTONDBLCLK and count < 4:
                pts1[count] = [x, y]
                count = count + 1
                cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
            elif count == 4:
                count = 5

        # create new window
        cv2.namedWindow('Original Image')
        cv2.setMouseCallback('Original Image', record_points)

        # refresh drawn img
        while count < 5:
            cv2.imshow('Original Image', img)
            cv2.waitKey(100)

        # Perspective Transform
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img_clear, M, (450, 450))

        cv2.imshow('Perspective Transformation', dst)

        # wait for user key interrupt and destrtoy window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(pts1)

    # END CLASS


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
