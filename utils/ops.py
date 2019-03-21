import cv2
import numpy as np
'''
For now, I'm using OpenCV library to provide this function
'''

_sharpening_kernel = [[-1, -1, -1],
                      [-1, 9, -1], 
                      [-1, -1, -1]]
                      
# 1. sharpness
def image_sharpenen(mat:np.ndarray):
    pass
# 2. Noise
def add_noise():
    pass
def remove_noise():
    pass
# 3. Dynamic range
def dynamic_range():
    pass
# 4. Tone reproduction
def get_image_tone():
    pass
# 5. Contrast
def add_contrast():
    pass
# 6. Color accuracy
def image_color_accuracy():
    pass
# 7. Distortion
def image_distortion():
    pass
# 8. Vignetting 
def image_vignetting():
    pass
# 9. Exposure accuracy
# 10. Lateral chromatic aberration
# 11. Lens flrae
# 12. Color moire
# 13. Artifacts