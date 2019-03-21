# import cv2
import numpy as np

def PSNR(base_image, another_image, R=255):
    if base_image.shape == 2:
        mse = MSE(base_image, another_image)
    elif base_image.shape == 3:
        mse = MSE_3_channel(base_image, another_image)
    return 10*np.log10(R**2/mse)

# TODO I don't know how this won't work well. All the test was failed
def SSIM_index(base_image, another_image, L=255):
    
    # Default values
    k1 = 0.01
    k2 = 0.03

    # print(np.iinfo(base_image.dtype.type).min, ' ', np.iinfo(base_image.dtype.type).max)

    mean_base = np.mean(base_image)
    mean_another = np.mean(another_image)
    variance_base = np.std(base_image, ddof=1)
    variance_another = np.std(another_image, ddof=1)
    covariance = correlation_coeeficient(base_image, another_image)
    C1 = (k1*L)**2
    C2 = (k2*L)**2

    num = (2*mean_base*mean_another+C1)*(2*covariance+C2)
    denum = ((mean_base**2)+(mean_another**2)+C1)*((variance_another**2)+(variance_base**2)+C2)
    print('num {} \ndenum {}'.format(num, denum))
    return num/denum

def MSE(image1, image2):
    assert image1.shape==image2.shape, 'These image has different shape'
    return (np.sum((image1-image2)**2))/image1.size

def MSE_3_channel(image1, image2):
    assert image1.shape==image2.shape, 'These image has different shape'
    ch1 = (np.sum((image1[:,:,0]-image2[:,:,0])**2))/np.prod(image1.shape[:2])
    ch2 = (np.sum((image1[:,:,1]-image2[:,:,1])**2))/np.prod(image1.shape[:2])
    ch3 = (np.sum((image1[:,:,2]-image2[:,:,2])**2))/np.prod(image1.shape[:2])
    return ch1+ch2+ch3/3

def correlation_coeeficient(base_image, another_image):
    mean_base = np.mean(base_image)
    mean_another = np.mean(another_image)
    corr = np.sum((xb-mean_base)*(xa-mean_another) for xb, xa in zip(base_image.flatten(), another_image.flatten()))
    return corr/base_image.size-1

'''

“Luck is what happens when preparation meets opportunity.” Seneca
'''