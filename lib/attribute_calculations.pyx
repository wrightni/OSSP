# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
import numpy as np
from scipy import stats as spstats
from ctypes import *


def analyze_srgb_image(input_image, watershed_image, segment_id=False):
    '''
    Cacluate the attributes for each segment given in watershed_image
    using the raw pixel values in input image. Attributes calculated for
    srgb type images.
    '''

    cdef int num_ws
    cdef int x_dim, y_dim, num_bands
    cdef int ws, b
    cdef int ws_size
    # cdef int histogram_i
    # cdef int histogram_e
   
    # If no segment id is provided, analyze the features for every watershed
    # in the input image. If a segment id is provided, just analyze the features
    # for that one segment.
    # We have to add +1 to num_ws because if the maximum value in watershed_image
    # is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
    if segment_id == False:
        num_ws = int(np.amax(watershed_image) + 1)
    else:
        num_ws = 1

    x_dim, y_dim, num_bands = np.shape(input_image)

    feature_matrix = np.empty((num_ws,16))
    cdef double [:, :] fm_view = feature_matrix

   #### Need to convert images to dtype c_int
    # input_image = np.ndarray.astype(input_image, c_int)
    # watershed_image = np.ndarray.astype(watershed_image, c_int)
    if segment_id is not False:
        internal, external = selective_pixel_sort(input_image, watershed_image,
                                    x_dim, y_dim, num_bands, segment_id)
    else:
        internal, external = pixel_sort(input_image, watershed_image,
                                        x_dim, y_dim,
                                        num_ws, num_bands)

    for ws in range(num_ws):

        ws_size = len(internal[0][ws])
        # Average Pixel Intensity of each band
        for b in range(3):
            fm_view[ws, b] = sum(internal[b][ws]) / float(ws_size)
            if fm_view[ws, b] < 1:
                fm_view[ws, b] = 1

        # Standard Deviation of each band
        fm_view[ws, 3] = np.std(internal[0][ws])
        fm_view[ws, 4] = np.std(internal[1][ws])
        fm_view[ws, 5] = np.std(internal[2][ws])

        # See Miao et al for band ratios
        # Band Ratio 1
        fm_view[ws, 6] = ((fm_view[ws, 2] - fm_view[ws, 0]) /
                       (fm_view[ws, 2] + fm_view[ws, 0]))
        # Band Ratio 2
        fm_view[ws, 7] = ((fm_view[ws, 2] - fm_view[ws, 1]) /
                       (fm_view[ws, 2] + fm_view[ws, 1]))
        # Band Ratio 3
        # Prevent division by 0
        if (2 * fm_view[ws, 2] - fm_view[ws, 1] - fm_view[ws, 0]) < 1:
            fm_view[ws, 8] = 0
        else:
            fm_view[ws, 8] = ((fm_view[ws, 1] - fm_view[ws, 0]) /
                           (2 * fm_view[ws, 2] - fm_view[ws, 1] - fm_view[ws, 0]))

        # Size of Superpixel
        fm_view[ws, 9] = ws_size

        # Entropy
        histogram_i = np.bincount(internal[1][ws])
        fm_view[ws, 10] = spstats.entropy(histogram_i, base=2)

        ## Neighborhood Values
        # N. Average Intensity
        fm_view[ws, 11] = sum(external[1][ws]) / float(len(external[1][ws]))
        # N. Standard Deviation
        fm_view[ws, 12] = np.std(external[1][ws])
        # N. Maximum Single Value
        fm_view[ws, 13] = np.amax(external[1][ws])
        # N. Entropy
        histogram_e = np.bincount(external[1][ws])
        fm_view[ws, 14] = spstats.entropy(histogram_e, base=2)

        # Date of image acquisition (removed, but need placeholder)
        fm_view[ws, 15] = 0

    feature_matrix = np.copy(fm_view)
    return feature_matrix


def analyze_ms_image(input_image, watershed_image, segment_id=False):
    '''
    Cacluate the attributes for each segment given in watershed_image
    using the raw pixel values in input image. Attributes calculated for
    multispectral type WorldView 2 images.
    '''
    # feature_matrix = []

    cdef int num_ws
    cdef int x_dim, y_dim, num_bands
    cdef double features[18]
    cdef int ws, b
    cdef int ws_size

    # If no segment id is provided, analyze the features for every watershed
    # in the input image. If a segment id is provided, just analyze the features
    # for that one segment.
    # We have to add +1 to num_ws because if the maximum value in watershed_image
    # is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
    if segment_id == False:
        num_ws = int(np.amax(watershed_image) + 1)
    else:
        num_ws = 1

    feature_matrix = np.empty((num_ws,18))
    cdef double [:, :] fm_view = feature_matrix

    x_dim, y_dim, num_bands = np.shape(input_image)

   #### Need to convert images to dtype c_int (done elsewhere)
    # input_image = np.ndarray.astype(input_image, c_int)
    # watershed_image = np.ndarray.astype(watershed_image, c_int)
    if segment_id is not False:
        internal, external = selective_pixel_sort(input_image, watershed_image,
                                    x_dim, y_dim, num_bands, segment_id)
    else:
        internal, external = pixel_sort(input_image, watershed_image,
                                        x_dim, y_dim,
                                        num_ws, num_bands)

    for ws in range(num_ws):

        ws_size = len(internal[0][ws])
        # Average Pixel Intensity of each band
        for b in range(8):
            fm_view[ws,b] = sum(internal[b][ws]) / float(ws_size)
            if fm_view[ws,b] < 1:
                fm_view[ws,b] = 1


        # Important band ratios
        fm_view[ws, 8] = fm_view[ws, 0] / fm_view[ws, 2]
        fm_view[ws, 9] = fm_view[ws, 1] / fm_view[ws, 6]
        fm_view[ws, 10] = fm_view[ws, 4] / fm_view[ws, 6]
        fm_view[ws, 11] = fm_view[ws, 3] / fm_view[ws, 5]
        fm_view[ws, 12] = fm_view[ws, 3] / fm_view[ws, 6]
        fm_view[ws, 13] = fm_view[ws, 3] / fm_view[ws, 7]
        fm_view[ws, 14] = fm_view[ws, 4] / fm_view[ws, 6]

        # N. Average Intensity
        fm_view[ws, 15] = sum(external[4][ws]) / float(len(external[4][ws]))

        # b1-b7 / b1+b7
        fm_view[ws, 16] = ((fm_view[ws, 0] - fm_view[ws, 6]) / (fm_view[ws, 0] + fm_view[ws, 6]))

        # b3-b5 / b3+b5
        fm_view[ws, 17] = ((fm_view[ws, 2] - fm_view[ws, 4]) / (fm_view[ws, 2] + fm_view[ws, 4]))

    feature_matrix = np.copy(fm_view)
    return feature_matrix


def analyze_pan_image(input_image, watershed_image, date, segment_id=False):
    '''
    Cacluate the attributes for each segment given in watershed_image
    using the raw pixel values in input image. Attributes calculated for
    srgb type images.
    '''
    feature_matrix = []

    cdef int num_ws
    cdef int x_dim, y_dim, num_bands
    cdef double features[12]
    cdef int ws, b
    # cdef int histogram_i
    # cdef int histogram_e
   
    # If no segment id is provided, analyze the features for every watershed
    # in the input image. If a segment id is provided, just analyze the features
    # for that one segment.
    # We have to add +1 to num_ws because if the maximum value in watershed_image
    # is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
    if segment_id == False:
        num_ws = int(np.amax(watershed_image) + 1)
    else:
        num_ws = 1

    x_dim, y_dim, num_bands = np.shape(input_image)

   #### Need to convert images to dtype c_int
    # input_image = np.ndarray.astype(input_image, c_int)
    # watershed_image = np.ndarray.astype(watershed_image, c_int)
    if segment_id is not False:
        internal, external = selective_pixel_sort(input_image, watershed_image,
                                    x_dim, y_dim, num_bands, segment_id)
    else:
        internal, external = pixel_sort(input_image, watershed_image,
                                        x_dim, y_dim,
                                        num_ws, num_bands)

    for ws in range(num_ws):
        
        # Check for empty watershed labels
        if internal[0][ws] == []:
            features = [0 for _ in range(12)]
            feature_matrix.append(features)
            continue

        # Average Pixel Intensity
        features[0] = np.average(internal[0][ws])
        if features[0] < 1:
            features[0] = 1

        # Median Pixel Value
        features[1] = np.median(internal[0][ws])
        # Segment Minimum
        features[2] = np.amin(internal[0][ws])
        # Segment Maximum
        features[3] = np.amax(internal[0][ws])
        # Standard Deviation
        features[4] = np.std(internal[0][ws])
        # Size
        features[5] = len(internal[0][ws])

        # Entropy
        histogram_i = np.bincount(internal[0][ws])
        features[6] = spstats.entropy(histogram_i, base=2)

        ## Neighborhood Values
        # N. Average Intensity
        features[7] = np.average(external[0][ws])
        # N. Standard Deviation
        features[8] = np.std(external[0][ws])
        # N. Maximum Single Value
        features[9] = np.amax(external[0][ws])
        # N. Entropy
        histogram_e = np.bincount(external[0][ws])
        features[10] = spstats.entropy(histogram_e, base=2)

        # Date of image acquisition
        features[11] = int(date)

        feature_matrix.append(features)

    return feature_matrix


def selective_pixel_sort(int[:,:,:] intensity_image_view,
                         int[:,:] label_image_view,
                         int x_dim, int y_dim, 
                         int num_bands, int label):
    
    cdef int y,x,i,w,b
    cdef int window[4]
    cdef int sn

    # Output variables. 
    #  Future work: Improve data structure here to something more efficient.
    internal = [[[]] for _ in range(num_bands)]
    external = [[[]] for _ in range(num_bands)]

    # Moving window that defines the neighboring region for each pixel
    window = [-4, -3, 3, 4]

    for y in range(y_dim):
        for x in range(x_dim):
            # Set the current segment number
            sn = label_image_view[x, y]

            # Select only the ws with the correct label
            if sn != label:
                continue

            # Assign the internal pixel
            for b in range(num_bands):
                internal[b][0].append(intensity_image_view[x, y, b])

            # Determine the external values within the window
            for w in range(4):
                i = window[w]
                # Determine the external values in the x-axis
                # Check for edge conditions
                if (x+i < 0) or (x+i >= x_dim):
                    continue
                if label_image_view[x+i, y] != sn:
                    for b in range(num_bands):
                        external[b][0].append(intensity_image_view[x+i, y, b])
                # Determine the external values in the y-axis
                # Check for edge conditions
                if (y+i < 0) or (y+i >= y_dim):
                    continue
                if label_image_view[x, y+i] != sn:
                    for b in range(num_bands):
                        external[b][0].append(intensity_image_view[x, y+i, b])

    return internal, external


def pixel_sort(int[:,:,:] intensity_image_view,
               int[:,:] label_image_view,
               int x_dim, int y_dim, int num_ws, int num_bands):
    '''
    Given an intensity image and label image of the same dimension, sort
    pixels into a list of internal and external intensity pixels for every 
    label in the label image. 
    Returns:
        Internal: Array of length (number of labels), each element is a list
            of intensity values for that label number.
        External: Array of length (number of labels), each element is a list
            of intensity values that are adjacent to that label number.
    '''
    cdef int y,x,i,w
    cdef int window[4]
    cdef int sn

    # Output variables. 
    #  Future work: Improve data structure here to something more efficient.
    internal = [[[] for _ in range(num_ws)] for _ in range(num_bands)]
    external = [[[] for _ in range(num_ws)] for _ in range(num_bands)]

    # internal = cvarray(shape=(num_ws,1), itemsize=sizeof(int), format="i")
    # cdef int [:] internal_view = internal
    
    # external = cvarray(shape=(num_ws,1), itemsize=sizeof(int), format="i")
    # cdef int [:] external_view = external

    # Moving window that defines the neighboring region for each pixel
    window = [-4, -3, 3, 4]

    for y in range(y_dim):
        for x in range(x_dim):
            # Ignore pixels whose value is 0 (no data)
            if intensity_image_view[x, y, 0] == 0:
                continue

            # Set the current segment number
            sn = label_image_view[x,y]
            # Assign the internal pixel
            for b in range(num_bands):
                internal[b][sn].append(intensity_image_view[x, y, b])

            # Determine the external values within the window
            for w in range(4):
                i = window[w]
                # Determine the external values in the x-axis
                # Check for edge conditions
                if (x+i < 0) or (x+i >= x_dim):
                    continue
                if label_image_view[x+i, y] != sn:
                    for b in range(num_bands):
                        external[b][sn].append(intensity_image_view[x+i, y, b])

                # Determine the external values in the y-axis
                # Check for edge conditions
                if (y+i < 0) or (y+i >= y_dim):
                    continue
                if label_image_view[x, y+i] != sn:
                    for b in range(num_bands):
                        external[b][sn].append(intensity_image_view[x, y+i, b])

    return internal, external