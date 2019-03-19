# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
import numpy as np
from scipy import stats as spstats
from ctypes import *
import psutil


def analyze_srgb_image(input_image, watershed_image, segment_id=False):
    '''
    Cacluate the attributes for each segment given in watershed_image
    using the raw pixel values in input image. Attributes calculated for
    srgb type images.
    '''
    cdef int num_ws
    cdef int x_dim, y_dim, num_bands
    cdef int ws, b, i
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

    num_bands, x_dim, y_dim = np.shape(input_image)

    feature_matrix = np.empty((num_ws,16), dtype=c_float)
    cdef float [:, :] fm_view = feature_matrix

    print(num_bands, x_dim, y_dim)
    #### Need to convert images to dtype c_int
    # input_image = np.ndarray.astype(input_image, c_int)
    # watershed_image = np.ndarray.astype(watershed_image, c_int)
    if segment_id is not False:
        internal, external = selective_pixel_sort(input_image, watershed_image,
                                                  x_dim, y_dim, num_bands, segment_id)
    else:
        internal, external, internal_ext, external_ext = pixel_sort_extended(input_image, watershed_image,
                                                                            x_dim, y_dim,
                                                                            num_ws, num_bands)

    print(external_ext[6000])
    print(last_index(external_ext[6000]))
    for ws in range(num_ws):
        # Average and Variance of Pixel Intensity for each band
        for b in range(3):
            count = internal[ws, b, 0]
            mean = internal[ws, b, 1]
            M2 = internal[ws, b, 2]
            variance = M2 / count
            if mean < 1:
                mean = 1
            fm_view[ws, b] = mean
            fm_view[ws, b+3] = variance

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
        fm_view[ws, 9] = internal[ws, 0, 0]

        # Entropy
        histogram_i = internal_ext[ws] #np.bincount(internal[1][ws])
        fm_view[ws, 10] = spstats.entropy(histogram_i, base=2)

        ## Neighborhood Values
        # N. Average Intensity
        n_mean = external[ws, 1, 1]
        fm_view[ws, 11] = n_mean
        # N. Standard Deviation
        n_var = external[ws, 1, 2] / external[ws, 1, 0]
        fm_view[ws, 12] = n_var
        # N. Maximum Single Value
        n_max = last_index(external_ext[ws])
        fm_view[ws, 13] = n_max
        # N. Entropy
        histogram_e = external_ext[ws]
        # histogram_e = np.bincount(external[1][ws])
        fm_view[ws, 14] = spstats.entropy(histogram_e, base=2)

        # Date of image acquisition (removed, but need placeholder)
        fm_view[ws, 15] = 0

        if 5999 < ws < 6020:
            print(feature_matrix[ws][0])
            # ws_size = len(internal_old[0][ws])
            # print(sum(internal_old[0][ws]) / float(ws_size))
            # print("-")

    return np.copy(fm_view)


def analyze_ms_image(input_image, watershed_image, segment_id=False):
    '''
    Cacluate the attributes for each segment given in watershed_image
    using the raw pixel values in input image. Attributes calculated for
    multispectral type WorldView 2 images.
    '''
    cdef int num_bands, x_dim, y_dim
    cdef int ws, b
    cdef int num_ws

    num_bands, x_dim, y_dim = np.shape(input_image)
    # If no segment id is provided, analyze the features for every watershed
    # in the input image. If a segment id is provided, just analyze the features
    # for that one segment.
    # We have to add +1 to num_ws because if the maximum value in watershed_image
    # is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
    if segment_id == False:
        num_ws = int(np.amax(watershed_image) + 1)
    else:
        num_ws = 1

    feature_matrix = np.empty((num_ws, 18), dtype=c_float)
    cdef float[:, :] fm_view = feature_matrix

    # Choose how to sort pixels based
    if segment_id is not False:
        internal, external = selective_pixel_sort(input_image, watershed_image,
                                    x_dim, y_dim, num_bands, segment_id)
    else:
        internal, external = pixel_sort(input_image, watershed_image,
                                    x_dim, y_dim, num_ws, num_bands)

    for ws in range(num_ws):
        # Average Pixel Intensity of each band
        for b in range(8):
            mean = internal[ws, b, 1]
            if mean < 1:
                mean = 1
            fm_view[ws, b] = mean

        # Important band ratios
        fm_view[ws, 8] = fm_view[ws, 0] / fm_view[ws, 2]
        fm_view[ws, 9] = fm_view[ws, 1] / fm_view[ws, 6]
        fm_view[ws, 10] = fm_view[ws, 4] / fm_view[ws, 6]
        fm_view[ws, 11] = fm_view[ws, 3] / fm_view[ws, 5]
        fm_view[ws, 12] = fm_view[ws, 3] / fm_view[ws, 6]
        fm_view[ws, 13] = fm_view[ws, 3] / fm_view[ws, 7]
        fm_view[ws, 14] = fm_view[ws, 4] / fm_view[ws, 6]

        # N. Average Intensity
        n_mean = external[ws, 4, 1]
        fm_view[ws, 15] = n_mean

        # b1-b7 / b1+b7
        fm_view[ws, 16] = ((fm_view[ws, 0] - fm_view[ws, 6]) / (fm_view[ws, 0] + fm_view[ws, 6]))

        # b3-b5 / b3+b5
        fm_view[ws, 17] = ((fm_view[ws, 2] - fm_view[ws, 4]) / (fm_view[ws, 2] + fm_view[ws, 4]))

    return np.copy(fm_view)


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


def selective_pixel_sort(unsigned char[:,:,:] intensity_image_view,
                         unsigned int[:,:] label_image_view,
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
                internal[b][0].append(intensity_image_view[b, x, y])

            # Determine the external values within the window
            for w in range(4):
                i = window[w]
                # Determine the external values in the x-axis
                # Check for edge conditions
                if (x+i < 0) or (x+i >= x_dim):
                    continue
                if label_image_view[x+i, y] != sn:
                    for b in range(num_bands):
                        external[b][0].append(intensity_image_view[b, x+i, y])
                # Determine the external values in the y-axis
                # Check for edge conditions
                if (y+i < 0) or (y+i >= y_dim):
                    continue
                if label_image_view[x, y+i] != sn:
                    for b in range(num_bands):
                        external[b][0].append(intensity_image_view[b, x, y+i])

    return internal, external


def pixel_sort(const unsigned char[:,:,:] intensity_image_view,
               const unsigned int[:,:] label_image_view,
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
    cdef int x, y, sn, i, w, b
    cdef unsigned char new_val
    cdef float count, mean, M2
    cdef float delta, delta2
    cdef char window[4]

    # Output variables.
    internal = np.zeros((num_ws, num_bands, 3), dtype=c_float)
    cdef float[:, :, :] in_view = internal
    external = np.zeros((num_ws, num_bands, 3), dtype=c_float)
    cdef float[:, :, :] ex_view = external

    # Moving window that defines the neighboring region for each pixel
    window = [-4, -3, 3, 4]

    for y in range(y_dim):
        for x in range(x_dim):
            # Ignore pixels whose value is 0 (no data)
            if intensity_image_view[0, x, y] == 0:
                continue

            # Set the current segment number
            sn = label_image_view[x, y]
            # Assign the internal pixel
            for b in range(num_bands):
                # Find the new pixel
                new_val = intensity_image_view[b, x, y]
                # Read the previous values
                count = in_view[sn, b, 0]
                mean = in_view[sn, b, 1]
                M2 = in_view[sn, b, 2]

                # Update the stored values
                count += 1
                delta = new_val - mean
                mean += delta / count
                delta2 = new_val - mean
                M2 += delta * delta2

                # Update the internal list
                in_view[sn, b, 0] = count
                in_view[sn, b, 1] = mean
                in_view[sn, b, 2] = M2

            # Determine the external values within the window
            for w in range(4):
                i = window[w]
                # Determine the external values in the x-axis
                # Check for edge conditions
                if (x + i < 0) or (x + i >= x_dim):
                    continue
                if label_image_view[x + i, y] != sn:
                    for b in range(num_bands):
                        new_val = intensity_image_view[b, x + i, y]
                        # Read the previous values
                        count = ex_view[sn, b, 0]
                        mean = ex_view[sn, b, 1]
                        M2 = ex_view[sn, b, 2]

                        # Update the stored values
                        count += 1
                        delta = new_val - mean
                        mean += delta / count
                        delta2 = new_val - mean
                        M2 += delta * delta2

                        # Update the internal list
                        ex_view[sn, b, 0] = count
                        ex_view[sn, b, 1] = mean
                        ex_view[sn, b, 2] = M2

                # Determine the external values in the y-axis
                # Check for edge conditions
                if (y + i < 0) or (y + i >= y_dim):
                    continue
                if label_image_view[x, y + i] != sn:
                    for b in range(num_bands):
                        new_val = intensity_image_view[b, x, y + i]
                        # Read the previous values
                        count = ex_view[sn, b, 0]
                        mean = ex_view[sn, b, 1]
                        M2 = ex_view[sn, b, 2]

                        # Update the stored values
                        count += 1
                        delta = new_val - mean
                        mean += delta / count
                        delta2 = new_val - mean
                        M2 += delta * delta2

                        # Update the internal list
                        ex_view[sn, b, 0] = count
                        ex_view[sn, b, 1] = mean
                        ex_view[sn, b, 2] = M2

    return internal, external


def pixel_sort_extended(const unsigned char[:,:,:] intensity_image_view,
                        const unsigned int[:,:] label_image_view,
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
    cdef int x, y, sn, i, w, b
    cdef int h_count
    cdef unsigned char new_val
    cdef float count, mean, M2
    cdef float delta, delta2
    cdef char window[4]

    # Output statistical variables.
    internal = np.zeros((num_ws, num_bands, 3), dtype=c_float)
    cdef float[:, :, :] in_view = internal
    external = np.zeros((num_ws, num_bands, 3), dtype=c_float)
    cdef float[:, :, :] ex_view = external

    # Output histogram of each segment
    internal_ext = np.zeros((num_ws, 256), dtype=c_int)
    cdef int[:, :] in_ext_view = internal_ext
    external_ext = np.zeros((num_ws, 256), dtype=c_int)
    cdef int[:, :] ex_ext_view = external_ext

    # Moving window that defines the neighboring region for each pixel
    window = [-4, -3, 3, 4]

    for y in range(y_dim):
        for x in range(x_dim):
            # Ignore pixels whose value is 0 (no data)
            if intensity_image_view[0, x, y] == 0:
                continue

            # Set the current segment number
            sn = label_image_view[x, y]
            # Assign the internal pixel
            for b in range(num_bands):
                # Find the new pixel
                new_val = intensity_image_view[b, x, y]
                # Read the previous values
                count = in_view[sn, b, 0]
                mean = in_view[sn, b, 1]
                M2 = in_view[sn, b, 2]

                # Update the stored values
                count += 1
                delta = new_val - mean
                mean += delta / count
                delta2 = new_val - mean
                M2 += delta * delta2

                # Update the internal list
                in_view[sn, b, 0] = count
                in_view[sn, b, 1] = mean
                in_view[sn, b, 2] = M2

                # Increment this pixel value in the b0 histogram
                if b == 0:
                    h_count = in_ext_view[sn, new_val]
                    h_count += 1
                    in_ext_view[sn, new_val] = h_count

            # Determine the external values within the window
            for w in range(4):
                i = window[w]
                # Determine the external values in the x-axis
                # Check for edge conditions
                if (x + i < 0) or (x + i >= x_dim):
                    continue
                if label_image_view[x + i, y] != sn:
                    for b in range(num_bands):
                        new_val = intensity_image_view[b, x + i, y]
                        # Read the previous values
                        count = ex_view[sn, b, 0]
                        mean = ex_view[sn, b, 1]
                        M2 = ex_view[sn, b, 2]

                        # Update the stored values
                        count += 1
                        delta = new_val - mean
                        mean += delta / count
                        delta2 = new_val - mean
                        M2 += delta * delta2

                        # Update the internal list
                        ex_view[sn, b, 0] = count
                        ex_view[sn, b, 1] = mean
                        ex_view[sn, b, 2] = M2

                        # Increment this pixel value in the b0 histogram
                        if b == 0:
                            h_count = ex_ext_view[sn, new_val]
                            h_count += 1
                            ex_ext_view[sn, new_val] = h_count

                # Determine the external values in the y-axis
                # Check for edge conditions
                if (y + i < 0) or (y + i >= y_dim):
                    continue
                if label_image_view[x, y + i] != sn:
                    for b in range(num_bands):
                        new_val = intensity_image_view[b, x, y + i]
                        # Read the previous values
                        count = ex_view[sn, b, 0]
                        mean = ex_view[sn, b, 1]
                        M2 = ex_view[sn, b, 2]

                        # Update the stored values
                        count += 1
                        delta = new_val - mean
                        mean += delta / count
                        delta2 = new_val - mean
                        M2 += delta * delta2

                        # Update the internal list
                        ex_view[sn, b, 0] = count
                        ex_view[sn, b, 1] = mean
                        ex_view[sn, b, 2] = M2

                        # Increment this pixel value in the b0 histogram
                        if b == 0:
                            h_count = ex_ext_view[sn, new_val]
                            h_count += 1
                            ex_ext_view[sn, new_val] = h_count

    return internal, external, internal_ext, external_ext


cdef int last_index(int[:] lst):
    cdef int i
    for i in range(255,-1,-1):
        if lst[i] != 0:
            return i

    return 0

def mem():
    print str(round(psutil.Process().memory_info().rss/1024./1024., 2)) + ' MB'


def pixel_sort_old(unsigned char[:,:,:] intensity_image_view,
               unsigned int[:,:] label_image_view,
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
        mem()
        for x in range(x_dim):
            # Ignore pixels whose value is 0 (no data)
            if intensity_image_view[x, y, 0] == 0:
                continue

            # Set the current segment number
            sn = label_image_view[x,y]
            # Assign the internal pixel
            for b in range(num_bands):
                internal[b][sn].append(intensity_image_view[b, x, y])
            # Determine the external values within the window
            for w in range(4):
                i = window[w]
                # Determine the external values in the x-axis
                # Check for edge conditions
                if (x+i < 0) or (x+i >= x_dim):
                    continue
                if label_image_view[x+i, y] != sn:
                    for b in range(num_bands):
                        external[b][sn].append(intensity_image_view[b, x+i, y])

                # Determine the external values in the y-axis
                # Check for edge conditions
                if (y+i < 0) or (y+i >= y_dim):
                    continue
                if label_image_view[x, y+i] != sn:
                    for b in range(num_bands):
                        external[b][sn].append(intensity_image_view[b, x, y+i])

    return internal

# From wikipedia: Welfords algorithm
# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
# cdef float update(float count, float mean, float M2, char newValue):
#     cdef float delta, delta2
#     count += 1
#     delta = newValue - mean
#     mean += delta / count
#     delta2 = newValue - mean
#     M2 += delta * delta2
#
#     return (count, mean, M2)

# retrieve the mean, variance and sample variance from an aggregate
# def finalize(float count, float M2):
#     cdef float variance sample_variance
#     if count < 2:
#         return 1
#
#     variance = M2 / count
#     sampleVariance = M2 / (count - 1)
#
#     return variance, sampleVariance