import cv2
import numpy as np
import math
from PIL import Image

from scipy.signal import convolve2d
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim




im = np.array (Image.open ('D:\chrome download\srgan-master\srgan-master\pic\\1.jpg'),'f')#将图像1数据转换为float型
im = cv2.resize(im, (200, 200), interpolation=cv2.INTER_CUBIC)
im2 = np.array (Image.open ('D:\chrome download\srgan-master\srgan-master\pic\\2.jpg'),'f')#将图像2数据转换为float型
im2= cv2.resize(im2, (200, 200), interpolation=cv2.INTER_CUBIC)
im22 = np.array (Image.open ('D:\Python\SRGAN_Wasserstein-master\SRGAN_Wasserstein-master\\111.jpg'),'f')#将图像2数据转换为float型
im22= cv2.resize(im22, (200, 200), interpolation=cv2.INTER_CUBIC)
im3 = np.array (Image.open ('D:\chrome download\srgan-master\srgan-master\pic\\3.jpg'),'f')#将图像2数据转换为float型
im3= cv2.resize(im3, (200, 200), interpolation=cv2.INTER_CUBIC)
im4 = np.array (Image.open ('D:\chrome download\srgan-master\srgan-master\pic\\4.jpg'),'f')#将图像2数据转换为float型
im4= cv2.resize(im4, (200, 200), interpolation=cv2.INTER_CUBIC)
im5 = np.array (Image.open ('D:\chrome download\srgan-master\srgan-master\pic\\5.jpg'),'f')#将图像2数据转换为float型
im5= cv2.resize(im5, (200, 200), interpolation=cv2.INTER_CUBIC)


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# def psnr1(img1, img2):
#     mse = np.mean(img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

import cv2
import datetime
# newimg=cv2.imread("D:\chrome download\srgan-master\srgan-master\pic\\1.jpg")
# now=datetime.datetime.now()
# lbimg=cv2.bilateralFilter(newimg,3,500,500)
# print('requests', (datetime.datetime.now() - now).microseconds)
# cv2.imshow('src',newimg)
# cv2.imshow('dst',lbimg)
# cv2.waitKey()
# cv2.destroyAllWindows()

def to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

if __name__ == '__main__':
    psnr1 = psnr(im,im2)
    # psnr11 = compare_psnr(im, im2)
    ssim1 = compare_ssim(to_gray(im),to_gray(im2),multichannel=False)
    psnr22 = psnr(im,im22)
    psnr2 = psnr(im, im3)
    psnr3 = psnr(im, im4)
    ssim3 = compare_ssim(to_gray(im),to_gray(im4),multichannel=False)
    psnr4 = psnr(im, im5)
    # psnr44 = compare_psnr(im, im5)
    ssim4 = compare_ssim(to_gray(im),to_gray(im5),multichannel=False)


    print('双线性：',psnr1)
    print('mse：',psnr2)
    print('感知损失：',psnr3)
    print('WGAN：',psnr4)
    print(psnr22)
    print(ssim1,ssim3,ssim4)
    # print(psnr11,psnr44)
