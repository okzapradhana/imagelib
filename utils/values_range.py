import numpy as np

'''
This code inspired by skimage to determine the range of color channel
by its dtype. Source: https://github.com/scikit-image/scikit-image/blob/master/skimage/util/dtype.py
'''

_integer_types = (np.byte, np.ubyte,            # 8 bits 
                  np.short, np.ushort,          # 16 bits
                  np.intc, np.uintc,            # maybe 16, 32 or 64 bits
                  np.int_, np.uint,             # 32 or 64 bits
                  np.longlong, np.ulonglong)    # 64 bits
                  # Unsigned means start with 0

_integer_ranges = {d: (np.iinfo(d).min, np.iinfo(d).max) for d in _integer_types}

dtype_ranges =  {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1)
}
dtype_ranges.update(_integer_ranges)

def color_ranges(mat, negative_values=False):
    vmin, vmax = dtype_ranges[mat.dtype.type]
    if negative_values:
        vmin = 0
    return vmin, vmax