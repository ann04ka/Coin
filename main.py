import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from o_v import CNN, classif

class Image:
    def __init__(self, img):
        self.img = img
        self.contours = None
        self.bin_img = None
        self.area_img = None
        self.masked_img = None

    def GetCircleArea(self):
        """ Displays to console the areas of fig on image
        Argument:
            img (array): Original image
        Returns:
            area_img (array): Binarized image with fig and their areas(px)
        """
        # cv.imwrite("Gray.jpg", self.img)
        (T, Otsu) = cv.threshold(self.img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        # cv.imwrite("Otsu.jpg", Otsu)
        # Режми поиска контуров RETR_EXTERNAL извлекает только крайние внешние контуры
        # Алгоритм аппроксимации контура CHAIN_APPROX_SIMPLE сжимает горизонтальные, вертикальные и диагональные сегменты, оставляя только их конечные точки.
        # Например, прямоугольный контур вверх-вправо кодируется четырьмя точками.
        self.contours, hierarchy = cv.findContours(Otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.bin_img = np.full(self.img.shape, 0, dtype=np.uint8)
        cv.drawContours(self.bin_img, self.contours, -1, 255, 1, cv.LINE_AA, hierarchy, 1)
        # cv.imwrite("Bin.jpg", self.bin_img)
        print(len(self.contours))
        for cnt in self.contours:
            M = cv.moments(cnt)
            print(M['m01'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                print(cv.contourArea(cnt))
                text = 'S = %d px' % (cv.contourArea(cnt))
                font = cv.FONT_HERSHEY_SIMPLEX
                org = (cx - 50, cy)
                self.area_img = cv.putText(self.bin_img, text, org, font, 0.5, 255, 1, cv.LINE_AA)
                return self.area_img, self.contours

    def GetMaskedImg(self):
        """ Leaves only what lies inside the contours of the image
        Argument:
            img (array): Original image,
            bin_img (array): Binarized image with contours
        Returns:
            area_img (array): Binarized image with fig and their areas(px)
        """
        self.bin_img = cv.fillPoly(self.bin_img, self.contours, 255)
        # self.img = cv.GaussianBlur(self.img, (3, 3), 0)
        self.masked_img = cv.bitwise_not(self.img, self.bin_img)
        # cv.imwrite('Masked.jpg', self.masked_img)
        return self.masked_img

    def GetBBox(self, contours):
        q = 0
        for cnt in contours:

            x, y, w, h = cv.boundingRect(cnt)
            Bbox = self.masked_img[y:y+h, x:x+w]
            q = q + 1
            kernel = np.ones((2, 2), np.uint8)
            ONES = cv.morphologyEx(Bbox, cv.MORPH_GRADIENT, kernel)
            (T, Otsu) = cv.threshold(ONES, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
            Bbox = cv.resize(Otsu, (64, 64), interpolation=cv.INTER_CUBIC)
            cv.imshow('%d_Bbox' %q, Bbox)
            cv.waitKey(0)
            # cv.imwrite('%d_Bbox.jpg' %q, Bbox)

coins = cv.imread('1111.png')
coins = cv.cvtColor(coins, cv.COLOR_BGR2GRAY)
img = Image(coins)
area_img, contours = img.GetCircleArea()
# cv.imwrite("Area.jpg", area_img)
# mask = img.GetMaskedImg()
img.GetBBox(contours)
cv.imshow("area_img", area_img)
cv.waitKey(0)

