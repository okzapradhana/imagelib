# from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import sys
sys.path.append('..')
from utils import metrics

# image = Image.open('gambar.jpg')
image = cv2.imread('gambar.jpg', 0)
assert type(image)==np.ndarray, 'Image not loaded'

def test_ssim_blur():
    # gray = ImageOps.grayscale(image)
    # img_blur = image.filter(ImageFilter.BLUR)
    img_blur = cv2.blur(image, (5,5))
    ssim_index = metrics.SSIM_index(image, img_blur)
    print('SSIM Index value: {}'.format(ssim_index))
    assert ssim_index.size == 1, 'Should be single value'

def test_ssim_same_image():
    ssim_index = metrics.SSIM_index(image, image)
    print('SSIM Index value: {}'.format(ssim_index))
    assert ssim_index == 1, 'It\'s should be return 1'

if __name__ == '__main__':
    test_ssim_same_image()
    test_ssim_blur()
    print('Test passed')

