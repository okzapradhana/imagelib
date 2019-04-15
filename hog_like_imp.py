import cv2
import numpy as np


# Create sliding window to extract
def get_patch(img:np.ndarray, region_size:int=64, displacement_size:int=50):
    assert len(img.shape) == 3, 'The image should be in 3-Color channel'
    rows, cols = img.shape[:2]
    
    # Looping and yielding a patch or region
    for row in range(0, rows, displacement_size):
        if row + region_size>rows:
            continue
        for column in range(0, cols, displacement_size):
            if column + region_size>cols:
                continue
            yield img[row:row+region_size, column:column+region_size, :]

# Add zero padding in not-so-correct way
def add_image_padding(image, iterasi):
    temp_rgb = np.zeros((image.shape[0]+(2*iterasi), image.shape[1]+(2*iterasi), 3))
    for channel in range(3):
        temp = image[:,:,channel].copy()
        for i in range(0, iterasi):
            zero_padding_row = np.zeros(temp.shape[0])
            temp = np.insert(temp, temp.shape[1], zero_padding_row, axis=1)
            zero_padding_column = np.zeros(temp.shape[1])
            temp = np.insert(temp, temp.shape[0], zero_padding_column, axis=0)
        temp_t = temp.T
        for i in range(0, iterasi):
            zero_padding_row = np.zeros(temp_t.shape[0])
            temp_t = np.insert(temp_t, temp_t.shape[1], zero_padding_row, axis=1)
            zero_padding_column = np.zeros(temp_t.shape[1])
            temp_t = np.insert(temp_t, temp_t.shape[0], zero_padding_column, axis=0)
        temp = temp_t.T
        temp_rgb[:,:,channel] = temp
    return temp_rgb

# Extracting gradient and orientation
def get_gradient_mat(patch):
    padded_patch = add_image_padding(patch, 1)
    temp = patch.copy()
    for row in range(patch.shape[0]):
        for column in range(patch.shape[1]):
            # Please fix this with lambda func or comprehension
            grad = (padded_patch[column+1, row+2,:]-padded_patch[column+1, row])**2 + (padded_patch[column+2, row+1,:]-padded_patch[column, row+2])**2
            grad = grad**0.5
            # print(grad)
            temp[column, row] = grad
    return temp

def get_orientation_mat(patch):
    padded_patch = add_image_padding(patch, 1)
    temp = patch.copy()
    for row in range(patch.shape[0]):
        for column in range(patch.shape[1]):
            y_value = padded_patch[column+1, row+2,:]-padded_patch[column+1, row]
            x_value = padded_patch[column+2, row+1,:]-padded_patch[column, row+2]
            if np.nonzero(x_value) and np.nonzero(y_value):
                orient = [0,0,0]
            else:
                orient = y_value/x_value
            orient = np.arctan(orient)*180/np.pi
            # Imputation: NaN to zero
            orient = [0 if np.isnan(i) else i for i in orient]
            # Change quadran III to I
            orient = [i+180 if x<0 else i for x,i in zip(x_value, orient)]
            # print(orient)
            temp[column, row] = orient
    return temp

def l2_distance(arr1, arr2):
    dist = sum((a1-a2)**2 for a1,a2 in zip(arr1, arr2))
    return dist**0.5

def l1_distance(arr1, arr2):
    return sum(abs(a1-a2) for a1, a2 in zip(arr1, arr2))

# Assign to histogram with n-bin
# This part is originated by author
def get_index_bin(bin_list, number):
    if number >=180:
        return len(bin_list)-1
    else:
        subs_list = np.array(bin_list) - number
        if any(subs_list) < 0:
            return list(subs_list).index(np.min(subs_list[subs_list>0]))-1
        else:
            return list(subs_list).index(np.min(subs_list[subs_list>0]))

def compute_n_histogram(grad_mat, orient_mat, n_bins):
    histogram = np.zeros((n_bins, 3))
    n_bin_range = [180*n/n_bins for n in range(1,n_bins+1)]
    for row in range(grad_mat.shape[0]):
        for column in range(grad_mat.shape[1]):
            for ch in range(3):
                idx = get_index_bin(n_bin_range, orient_mat[row,column, ch])
                histogram[idx, ch] = grad_mat[row, column, ch]
    return histogram


# Executing
img = cv2.imread('zebra.jpg')
histograms1 = []
for patch in get_patch(img):
    grad_mat = get_gradient_mat(patch)
    orient_mat = get_orientation_mat(patch)
    histograms1.append(compute_n_histogram(grad_mat, orient_mat, 9))
print('done img1')
histograms1 = [item for items in histograms1 for item in items]
# histograms1 = reduce(operator.add, histograms1)
# histograms1 = lambda l: [item for sublist in histograms1 for item in sublist]

img = cv2.imread('zebra2.jpg')
histograms2 = []
for patch in get_patch(img):
    grad_mat = get_gradient_mat(patch)
    orient_mat = get_orientation_mat(patch)
    histograms2.append(compute_n_histogram(grad_mat, orient_mat, 9))
print('done img2')
# histograms2 = lambda l: [item for sublist in histograms2 for item in sublist]
histograms2 = [item for items in histograms2 for item in items]


img = cv2.imread('zebra3.jpg')
histograms3 = []
for patch in get_patch(img):
    grad_mat = get_gradient_mat(patch)
    orient_mat = get_orientation_mat(patch)
    histograms3.append(compute_n_histogram(grad_mat, orient_mat, 9))
print('done img3')
# histograms3 = lambda l: [item for sublist in histograms3 for item in sublist]
histograms3 = [item for items in histograms3 for item in items]


# You can change it to L1 distance
dist1_2 = l2_distance(histograms1, histograms2)
dist1_3 = l2_distance(histograms1, histograms3)
dist2_3 = l2_distance(histograms2, histograms3)

# You could improve this part
print('Distance from image 1 to 2 {}'.format(dist1_2))
print('Distance from image 1 to 3 {}'.format(dist1_3))
print('Distance from image 2 to 3 {}'.format(dist2_3))

