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
    cdef int ws, b, i, sid
    cdef int ws_size
   
    # If no segment id is provided, analyze the features for every watershed
    # in the input image. If a segment id is provided, just analyze the features
    # for that one segment.
    # We have to add +1 to num_ws because if the maximum value in watershed_image
    # is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
    if segment_id == False:
        num_ws = int(np.amax(watershed_image) + 1)
        sid = 0
    else:
        num_ws = 1
        sid = segment_id

    num_bands, x_dim, y_dim = np.shape(input_image)

    feature_matrix = np.zeros((num_ws,16), dtype=c_float)
    cdef float [:, :] fm_view = feature_matrix

    internal, external, internal_ext, external_ext = pixel_sort_extended(input_image, watershed_image,
                                                                         sid, x_dim, y_dim,
                                                                         num_ws, num_bands)

    for ws in range(num_ws):
        # If there are no pixels associated with this watershed, skip this iteration
        if internal[ws, 0, 0] < 1:
            continue

        # Average and Variance of Pixel Intensity for each band
        for b in range(3):
            count = internal[ws, b, 0]
            mean = internal[ws, b, 1]
            M2 = internal[ws, b, 2]
            variance = M2 / count
            if mean < 1:
                mean = 1
            fm_view[ws, b] = mean
            fm_view[ws, b+3] = variance**(1./2)

        # See Miao et al for band ratios
        # Division by zero is not possible because fm_view[ws,0:3] have a forced min of 1
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
        histogram_i = internal_ext[ws]#[:last_index(internal_ext[ws])] #np.bincount(internal[1][ws])
        fm_view[ws, 10] = spstats.entropy(histogram_i, base=2)

        # If there are no external pixels (usually when whole images is black)
        # skip assigning these values.
        if external[ws, 0, 0] < 1:
            continue
        ## Neighborhood Values
        # N. Average Intensity
        n_mean = external[ws, 1, 1]
        fm_view[ws, 11] = n_mean
        # N. Standard Deviation
        n_var = (external[ws, 1, 2] / external[ws, 1, 0])**(1./2)
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

    return np.copy(fm_view)


def analyze_ms_image(input_image, watershed_image, wb_ref, bp_ref, segment_id=False):
    '''
    Cacluate the attributes for each segment given in watershed_image
    using the raw pixel values in input image. Attributes calculated for
    multispectral type WorldView 2 images.
    '''
    cdef int num_bands, x_dim, y_dim
    cdef int ws, b, sid
    cdef int num_ws
    cdef double mean, n_mean, M2, variance, count
    cdef double wb_point, wb_rel, bp_point, bp_rel

    # If no segment id was given, set sn to zero to signal pixel_sort that
    #   all segments should be analyzed. Otherwise only the segment with
    #   the number == sn will be processed
    # We have to add +1 to num_ws because if the maximum value in watershed_image
    # is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
    if segment_id == False:
        num_ws = int(np.amax(watershed_image) + 1)
        sid = 0
    else:
        num_ws = 1
        sid = segment_id

    num_bands, x_dim, y_dim = np.shape(input_image)

    internal, external, internal_ext, external_ext = pixel_sort_extended(input_image, watershed_image,
                                                                         sid, x_dim, y_dim,
                                                                         num_ws, num_bands)

    feature_matrix = np.zeros((num_ws, 31), dtype=c_float)
    cdef float[:, :] fm_view = feature_matrix
    cdef float[:, :, :] in_view = internal
    cdef float[:, :, :] ex_view = external

    # wb_ref = wb_ref
    # for b in range(8):
    #     wb_ref[b] = wb_reference[b]

    for ws in range(num_ws):
        # If there are no pixels associated with this watershed, skip this iteration
        if in_view[ws, 0, 0] < 1:
            continue

        # Average Pixel Intensity of each band
        for b in range(8):
            count = in_view[ws, b, 0]
            mean = in_view[ws, b, 1]            
            if mean < 1:
                mean = 1
            fm_view[ws, b] = mean
        
        # Variance of band 7 (emperically the most useful)
        count = in_view[ws, 6, 0]
        M2 = in_view[ws, 6, 2]
        variance = M2 / count
        fm_view[ws, 8] = variance**(1./2) #11 14 15 13 8 9 12 10

        # Important band ratios
        fm_view[ws, 9] = fm_view[ws, 0] / fm_view[ws, 2]
        fm_view[ws, 10] = fm_view[ws, 1] / fm_view[ws, 6] #
        #fm_view[ws, 18] = fm_view[ws, 4] / fm_view[ws, 6] #
        #fm_view[ws, 19] = fm_view[ws, 3] / fm_view[ws, 5] #
        fm_view[ws, 11] = fm_view[ws, 3] / fm_view[ws, 6]
        #fm_view[ws, 21] = fm_view[ws, 3] / fm_view[ws, 7] #
        #fm_view[ws, 22] = fm_view[ws, 4] / fm_view[ws, 6] #

        # If there are no external pixels (usually when whole images is black)
        # skip assigning this value.
        if ex_view[ws, 4, 0] >= 1:
            # N. Average Intensity
            n_mean = ex_view[ws, 3, 1]
            fm_view[ws, 12] = n_mean
            n_mean = ex_view[ws, 7, 1]
            fm_view[ws, 13] = n_mean

        # b1-b7 / b1+b7
        fm_view[ws, 14] = ((fm_view[ws, 0] - fm_view[ws, 6]) / (fm_view[ws, 0] + fm_view[ws, 6]))

        # b3-b5 / b3+b5
        fm_view[ws, 15] = ((fm_view[ws, 2] - fm_view[ws, 4]) / (fm_view[ws, 2] + fm_view[ws, 4]))

        # Relative to the white balance point (b8 ignored emperically)
        for b in range(7):
            wb_point = wb_ref[b]
            wb_rel = fm_view[ws, b] / wb_point
            fm_view[ws, 16+b] = wb_rel #35

        # Relative to the dark reference point
        for b in range(8):
            bp_point = bp_ref[b]
            bp_rel = fm_view[ws, b] / bp_point
            fm_view[ws, 23+b] = bp_rel #35

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
    cdef int ws, b, sid
   
    # If no segment id was given, set sn to zero to signal pixel_sort that
    #   all segments should be analyzed. Otherwise only the segment with
    #   the number == sn will be processed
    # We have to add +1 to num_ws because if the maximum value in watershed_image
    # is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1

    if segment_id == False:
        num_ws = int(np.amax(watershed_image) + 1)
        sid = 0
    else:
        num_ws = 1
        sid = segment_id

    x_dim, y_dim, num_bands = np.shape(input_image)

    internal, external = pixel_sort(input_image, watershed_image,
                                    sid, x_dim, y_dim,
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


def pixel_sort(const unsigned char[:,:,:] intensity_image_view,
               const unsigned int[:,:] label_image_view,
               unsigned int segment_id, int x_dim, int y_dim, int num_ws, int num_bands):
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
    cdef int x, y, i, w, b
    cdef unsigned int sn
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

            # If a segment_id was given
            # Select only the ws with the correct label.
            #    set sn to zero to index in_view properly
            if segment_id != 0:
                if segment_id == sn:
                    sn = 0
                else:
                    continue

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
                        unsigned int segment_id, int x_dim, int y_dim, int num_ws, int num_bands):
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
    cdef int x, y, i, w, b
    cdef int h_count
    cdef unsigned int sn
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

            # If a segment_id was given
            # Select only the ws with the correct label.
            #    set sn to zero to index properly
            if segment_id != 0:
                if segment_id == sn:
                    sn = 0
                else:
                    continue

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
                if b == 1:
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
                        if b == 1:
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
                        if b == 1:
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