# title: Watershed Transform
# author: Nick Wright
# adapted from: Justin Chen, Arnold Song

import numpy as np
import gc
import warnings
from skimage import filters, morphology, feature, exposure, img_as_ubyte
from scipy import ndimage
from ctypes import *
from lib import rescale_intensity
from lib import utils

# For Testing:
from skimage import segmentation
import matplotlib.image as mimg


def segment_image(input_data, image_type=False):
    '''
    Wrapper function that handles all of the processing to create watersheds
    '''
 
    #### Define segmentation parameters
    # Sobel_threshold: Gradient values below this threshold will be set to zero
    # Amplification factor: Amount to scale gradient values by after threshold has been applied
    # Gauss_sigma: sigma value to use in the gaussian blur applied to the image prior to segmentation.
    #   Value chosen here should be based on the quality and resolution of the image
    # Feature_separation: minimum distance, in pixels, between the center point of multiple features. Use a lower value
    #   for lower resolution (.5m) images, and higher resolution for aerial images (~.1m).
    # These values are dependent on the type of imagery being processed, and are
    #   mostly empirically derived.
    # band_list contains the three bands to be used for segmentation
    if image_type == 'pan':
        sobel_threshold = 0.1
        amplification_factor = 2.
        gauss_sigma = 1
        feature_separation = 1
        band_list = [0, 0, 0]
    elif image_type == 'wv02_ms':
        sobel_threshold = 0.06
        amplification_factor = 2.5
        gauss_sigma = 1.5
        feature_separation = 1
        band_list = [4, 2, 1]
    else:   #image_type == 'srgb'
        sobel_threshold = 0.03
        amplification_factor = 3.1
        gauss_sigma = 2
        feature_separation = 5
        band_list = [0, 1, 2]

    segmented_data = watershed_transformation(input_data, band_list, sobel_threshold, amplification_factor,
                                              gauss_sigma,feature_separation)

    # Method that provides the user an option to view the original image
    #  side by side with the segmented image.
    # print(np.amax(segmented_data))
    image_data = np.array([input_data[band_list[0]],
                           input_data[band_list[1]],
                           input_data[band_list[2]]],
                          dtype=np.uint8)
    ws_bound = segmentation.find_boundaries(segmented_data)
    ws_display = utils.create_composite(image_data)

    save_name = '/Users/nicholas/Desktop/original_{}.png'
    mimg.imsave(save_name.format(np.random.randint(0,100)), ws_display, format='png')

    ws_display[:, :, 0][ws_bound] = 240
    ws_display[:, :, 1][ws_bound] = 80
    ws_display[:, :, 2][ws_bound] = 80

    save_name = '/Users/nicholas/Desktop/seg_{}.png'
    mimg.imsave(save_name.format(np.random.randint(0,100)), ws_display, format='png')

    return segmented_data


def watershed_transformation(image_data, band_list, sobel_threshold, amplification_factor, gauss_sigma, feature_separation):
    '''
    Runs a watershed transform on the main dataset
        1. Create a gradient image using the sobel algorithm
        2. Adjust the gradient image based on given threshold and amplification.
        3. Find the local minimum gradient values and place a marker
        4. Construct watersheds on top of the gradient image starting at the
            markers.
    '''
    # If this block has no data, return a placeholder watershed.
    if np.amax(image_data[0]) <= 1:
        # We just need the dimensions from one band
        return np.zeros(np.shape(image_data[0]))

    # Find the locations that contain no spectral data
    #  i.e. pixels that are 0 in all bands
    # empty_pixels = np.empty_like(image_data[0], dtype=np.bool8)
    # empty_pixels[(image_data[0] == 0)
    #              & (image_data[1] == 0)
    #              & (image_data[2] == 0)] = True

    edge_image = edge_detect(image_data, band_list, gauss_sigma, amplification_factor, sobel_threshold)
    image_data = None

    # Find local minimum values in the sobel image by inverting
    #   sobel_image and finding the local maximum values
    inv_edge = np.empty_like(edge_image, dtype=np.uint8)
    np.subtract(255, edge_image, out=inv_edge)

    local_min = feature.peak_local_max(inv_edge, min_distance=feature_separation,
                                       indices=False, num_peaks_per_label=1)
    inv_edge = None
    markers = ndimage.label(local_min)[0]
    local_min = None

    # Build a watershed from the markers on top of the edge image
    im_watersheds = morphology.watershed(edge_image, markers)
    # Clear gradient image data
    edge_image = None

    # Set all values outside of the image area (empty pixels, usually caused by
    #   orthorectification) to one value, at the end of the watershed list.
    # im_watersheds[empty_pixels] = np.amax(im_watersheds)+1
    gc.collect()
    return im_watersheds


def edge_detect(image_data, band_list, gauss_sigma, amplification_factor, sobel_threshold):

    # Apply a smoothing filter to remove noise and improve segmentation
    smooth_im_blue = ndimage.filters.gaussian_filter(image_data[band_list[2]], sigma=gauss_sigma)
    # Create a gradient image using a sobel filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sobel_image_blue = img_as_ubyte(filters.scharr(smooth_im_blue))
    smooth_im_blue = None

    # Do the same for the red band
    smooth_im_red = ndimage.filters.gaussian_filter(image_data[band_list[0]], sigma=gauss_sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sobel_image_red = img_as_ubyte(filters.scharr(smooth_im_red))
    smooth_im_red = None
    image_data = None

    # Find the absolute difference between the gradients in the blue and red image
    diff_image = np.empty_like(sobel_image_blue, dtype=np.int8)
    np.subtract(sobel_image_blue, sobel_image_red, out=diff_image)
    np.abs(diff_image, out=diff_image)
    sobel_image_red = None

    # Add the differences to the blue gradient image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff_image = img_as_ubyte(diff_image)
    sobel_image = np.empty_like(sobel_image_blue, dtype=np.uint8)
    np.add(sobel_image_blue, diff_image, out=sobel_image)

    sobel_image_blue = None
    diff_image = None

    # Adjust the sobel image based on the given threshold and amp factor.
    upper_threshold = 255. / amplification_factor
    if upper_threshold < 50:
        upper_threshold = 50

    sobel_image = rescale_intensity.rescale_intensity(sobel_image, 0, upper_threshold, 0, 255)
    # sobel_image = exposure.rescale_intensity(sobel_image,
    #                                          in_range=(0, upper_threshold),
    #                                          out_range=np.uint8)

    # Prevent the watersheds from 'leaking' along the sides of the image
    sobel_image[:, 0] = 255
    sobel_image[:, -1] = 255
    sobel_image[0, :] = 255
    sobel_image[-1, :] = 255

    # Set all values in the sobel image that are lower than the
    #   given threshold to zero.
    sobel_threshold *= 255
    sobel_image[sobel_image <= sobel_threshold] = 0

    gc.collect()
    return sobel_image