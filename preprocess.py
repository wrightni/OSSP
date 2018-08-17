# Code for preprocessing imagery for analysis by the OSSP image processing
# algorithm. Includes methods to split imagery into more manageable sections,
# and methods for histogram stretching to scale input images to the full 0,255
# range.
# Nicholas Wright
# 11/30/17

import argparse
import os
import math
import h5py
import numpy as np
import matplotlib.image as mimg
from skimage import exposure
from skimage.measure import block_reduce
import gdal
from lib import utils


def prepare_image(input_path, image_name, image_type,
                  output_path=None, number_of_splits=1, apply_correction=True, verbose=False):
    """
    Reads an image file and prepares it for further processing.
        1. Rescales image intensity based on the input histogram.
        2. Splits it into a N parts where N is the nearest perfect square
            to number_of_splits. These are saved to disk if N>1.
        3. Grids each split into ~600x600 pixel blocks. Images are processed one
            block at a time in Watershed and RandomForest. Blocks are in the
            format of [[[block1]], [[block2]], ...[[blockn]]], read left to
            right and top to bottom
    Image splits are saved as .h5 formatted datasets.
        Attrs: {[Image Date], [Image Type], [Dimensions]}
        Datasets: 1 for each image band, array of ~600x600 blocks
    If number of splits is 1, data is returned as a list of the band subimages
        and nothing is saved to disk.
    """

    # Open dataset with GDAL
    full_image_name = os.path.join(input_path, image_name)
    # Check to make sure the given file exists
    if os.path.isfile(full_image_name):
        if verbose:
            print("Reading image...")
        dataset = gdal.Open(full_image_name)
    else:
        print "File not found: " + full_image_name
        return None, None

    # Check number_of_splits input
    #   Round number_of_splits to the nearest square number
    base = round(math.sqrt(number_of_splits))
    # Approximate size of image block.
    # Might want to select this based on input image type?
    desired_block_size = 8000

    # Make sure each split is big enough for at least 2 grids in each 
    #   dimension. If not, there's really little reason to be splitting this 
    #   image in the first place.
    if number_of_splits < 1:
        while (dataset.RasterXSize / base < desired_block_size * 2
               or dataset.RasterYSize / base < desired_block_size * 2):
            base -= 1
            if verbose:
                print("Too many splits chosen, reducing to: %i" % (base * base))

    # Calculate grids for dividing raw image into chunks.
    #   Determine the number of splits in x and y dimensions. These are equal to
    #       eachother, but set up to allow non-square dimensions if desired.
    x_splits = int(base)
    y_splits = x_splits
    # Update number of splits
    number_of_splits = x_splits * y_splits

    # Determine the dimensions of each image split
    split_cols, split_rows, = find_splitsize(dataset.RasterXSize,
                                             dataset.RasterYSize,
                                             x_splits,
                                             y_splits)
    # Determine the block size within each split
    block_cols, block_rows = find_blocksize(split_cols,
                                            split_rows,
                                            desired_block_size)
    if verbose:
        print("Image Dimensions: %i x %i" % (dataset.RasterXSize,
                                             dataset.RasterYSize))
        print("Split Dimensions: %i x %i" % (split_cols, split_rows))
        print("Block Dimensions: %i x %i" % (block_cols, block_rows))

    # Find image metadata
    #   Pull the image date from the header information. Read_metadata assume
    #       early june date if no data can be found.
    #   Determine bands from dataset
    metadata = dataset.GetMetadata()
    image_date = read_metadata(metadata, image_type)
    band_count = dataset.RasterCount
    if verbose:
        print("Number of Bands: %i" % band_count)
        print("Using %i as image date." % image_date)

    # Verify output directory
    #   If no output directory was provided, default to input_dir/splits
    if output_path is None and number_of_splits > 1:
        output_path = os.path.join(input_path, "splits")
    # If the output path doesnt already exist, create that directory
    if output_path is not None:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    # Split and Grid Image
    #   Splits the image into number_of_split parts, and divides each split into
    #       block_cols X block_rows grids.
    #   If number_of_splits > 1, saves each split in its own .h5 file with
    #       n datasets; one for each spectral band.

    # Set the percentile thresholds at a temporary value until finding the 
    #   appropriate ones considering all three bands.
    lower = -1
    upper = -1

    # First for loop finds the threshold based on all bands
    for b in range(1, band_count + 1):
        # Flag for choosing whether to apply correction
        if apply_correction is False:
            continue

        if verbose:
            print("Analyzing band %s data..." % b)
        # Read the band information from the gdal dataset
        band = dataset.GetRasterBand(b)

        # Find the min and max image values
        bmin, bmax = band.ComputeRasterMinMax()

        # Determine the histogram using gdal
        nbins = int(bmax - bmin)
        hist = band.GetHistogram(bmin, bmax, nbins, approx_ok=0)
        bin_centers = range(int(bmin), int(bmax))
        bin_centers = np.array(bin_centers)

        # Remove the image data from memory for now
        band = None

        # Find the strongest (3) peaks in the band histogram
        peaks = find_peaks(hist, bin_centers, image_type)
        # Find the high and low threshold for rescaling image intensity
        lower_b, upper_b = find_threshold(hist, bin_centers,
                                          peaks, image_type)

        # For sRGB we want to scale each band by the min and max of all 
        #   bands. Check thresholds found for this band against any that
        #   have been previously found, and adjust if necessary. 
        if lower_b < lower or lower == -1:
            lower = lower_b
        if upper_b > upper or upper == -1:
            upper = upper_b

    # Now that we've checked the histograms of each band in the image,
    #   we can rescale and save each band.
    bands_output = {}  # {band_id: [subimage][row][column]}l
    for b in range(1, band_count + 1):
        # Read the gdal dataset and load into numpy array
        gdal_band = dataset.GetRasterBand(b)
        band = gdal_band.ReadAsArray()
        gdal_band = None

        if apply_correction is True:
            if verbose:
                print("Rescaling band %s" % b)
            # Rescale the band based on the lower and upper thresholds found
            band = rescale_band(band, lower, upper)
        else:
            if image_type == 'wv02_ms' or image_type == 'pan':
                band = rescale_band(band, 1, 2047)

        # If the image is not being split, construct image blocks and
        #   compile that data to return
        if number_of_splits == 1:
            bands_output[b], dimensions = construct_blocks(
                band,
                block_cols,
                block_rows,
                [split_rows, split_cols])
            if output_path is not None:
                # Saves the preprocessed image. Mostly used for
                # making the images for training set creation.
                fname = os.path.splitext(image_name)[0] + "_prep.h5"
                dst_file = os.path.join(output_path, fname)
                # Save the data to disk
                write_to_hdf5(dst_file, bands_output[b], b, image_type,
                              image_date, dimensions)
        else:
            if verbose:
                print("Splitting band %s..." % b)
            # Divide the data into a list of splits1
            band_split = split_band(band, x_splits,
                                    y_splits, split_cols, split_rows)
            snum = 1  # Tracker for file naming
            for split in band_split:
                # Grid this split into the appropriate number of blocks
                split_blocked, dimensions = construct_blocks(
                    split,
                    block_cols,
                    block_rows,
                    [split_rows, split_cols])
                # Determine the output filename
                fname = (os.path.splitext(image_name)[0]
                         + "_s{0:02d}of{1:02d}.h5".format(snum,
                                                          number_of_splits))
                dst_file = os.path.join(output_path, fname)
                # Save the data to disk
                write_to_hdf5(dst_file, split_blocked, b, image_type,
                              image_date, dimensions)
                snum += 1
        if verbose:
            print("Band %s complete" % b)

    meta_data = [dimensions, image_date]

    # If number_of_splits is more than 1, bands_output will be empty
    return bands_output, meta_data


def find_splitsize(total_cols, total_rows, col_splits, row_splits):
    """
    Determines the appropriate x (col) and y (row) dimensions for each
        image split. Total image size is rounded up to the nearest multiple
        of 100*#columns. This allows for easier creation of uniform image
        blocks, but the images will need to be padded with zeros to fit the
        increased size.
    """
    divisor = 100 * col_splits
    # Image dimension is rounded up (padded) to nearest 100*col_spits
    cols_pad = math.ceil(float(total_cols) / divisor) * divisor
    rows_pad = math.ceil(float(total_rows) / divisor) * divisor
    # Number of columns and rows in each split
    split_cols = int(cols_pad / col_splits)
    split_rows = int(rows_pad / row_splits)

    return split_cols, split_rows


def find_blocksize(x_dim, y_dim, desired_size):
    """
    Finds the appropriate block size for an input image of a given dimensions.
    Method returns the first factor of the input dimension that is greater than
        the desired size.
    """
    # Just in case x_dim and y_dim are smaller than expected
    if x_dim < desired_size or y_dim < desired_size:
        block_x = x_dim
        block_y = y_dim

    factors_x = factor(x_dim)
    factors_y = factor(y_dim)
    for x in factors_x:
        if x >= desired_size:
            block_x = x
            break
    for y in factors_y:
        if y >= desired_size:
            block_y = y
            break

    return int(block_x), int(block_y)


def factor(number):
    """
    Returns a sorted list of all of the factors of number using trial division.
    source: http://www.calculatorsoup.com/calculators/math/factors.php
    """
    factors = []
    s = int(math.ceil(math.sqrt(number)))

    for i in range(1, s):
        c = float(number) / i
        if int(c) == c:
            factors.append(c)
            factors.append(number / c)
    factors.sort()

    return factors


def read_metadata(metadata, image_type):
    """
    Parse image metadata information to find date.
    If image date cannot be found, return mean date of melt season (June1).
    This is likely to have to least impact on decision tree outcomes, as there
        will be less bias towards no melt vs lots of melt, but this should be tested.
        If 0 were to be used, then the decision tree would see that as very early
        season, since date is a numeric feature, and not a categorical one.
    """
    try:
        if image_type == 'srgb':
            header_date = metadata['EXIF_DateTime']
            image_date = header_date[5:7] + header_date[8:10]
        elif image_type == 'pan' or image_type == 'wv02_ms':
            # image_date = metadata['NITF_STDIDC_ACQUISITION_DATE'][4:8]
            image_date = metadata['NITF_IDATIM'][4:8]
    except KeyError:
        image_date = "0601"  # June first

    image_date = int(image_date)
    return image_date


def find_peaks(hist, bin_centers, image_type):
    """
    Finds the three strongest peaks in a given band.
    Criteria for each peak:
        Distance to the nearest neighboring peak is greater than one third the approx. dynamic range of the input image
        Has a minimum number of pixels in that peak, loosely based on image size
        Is greater than the directly adjacent bins, and the bins +/- 5 away
    """

    # Roughly define the smallest acceptable size of a peak based on the input image type.
    if image_type == 'srgb':
        min_count = 1000
    else:
        min_count = 100000
    # First find all potential peaks in the histogram
    peaks = []
    for i in range(1, len(bin_centers) - 1):
        # Acceptable width of peak is +/-5, except in edge cases
        if i < 5:
            w_l = i
            w_u = 5
        elif i > len(bin_centers) - 6:
            w_l = 5
            w_u = len(bin_centers) - 1 - i
        else:
            w_l = 5
            w_u = 5
        # Check neighboring peaks
        if (hist[i] >= hist[i + 1] and hist[i] >= hist[i - 1]
                and hist[i] >= hist[i - w_l] and hist[i] >= hist[i + w_u]):
            if hist[i] > min_count:
                peaks.append(bin_centers[i])

    num_peaks = len(peaks)
    distance = 5  # Initial distance threshold
    # One third the 'dynamic range' (radius from peak)
    distance_threshold = int((peaks[-1] - peaks[0]) / 6)
    # Min threshold
    if distance_threshold <= 5:
        distance_threshold = 5
    # Looking for three main peaks corresponding to the main surface types: 
    #   open water, MP and snow/ice
    # But any peak that passes the criteria is fine.
    while distance <= distance_threshold:
        i = 0
        to_remove = []
        # Cycle through all of the peaks
        while i < num_peaks - 1:
            # Check the current peak against the adjacent one. If they are closer 
            #   than the threshold distance, delete the lower valued peak
            if peaks[i + 1] - peaks[i] < distance:
                if (hist[np.where(bin_centers == peaks[i])[0][0]]
                        < hist[np.where(bin_centers == peaks[i + 1])[0][0]]):
                    to_remove.append(peaks[i])
                else:
                    to_remove.append(peaks[i + 1])
                    # Because we don't need to check the next peak again:
                    i += 1
            i += 1

        # Remove all of the peaks that did not meet the criteria above
        for j in to_remove:
            peaks.remove(j)

        # Recalculate the number of peaks left, and increase the distance threshold
        num_peaks = len(peaks)
        distance += 5

    return peaks


def find_threshold(hist, bin_centers, peaks, image_type):
    """
    Finds the upper and lower threshold for histogram stretching.
    Using the indices of the highest and lowest peak (by intensity, not # of pixels), this searches for an upper
    threshold that is both greater than the highest peak and has fewer than 10% the number of pixels, and a lower
    threshold that is both less than the lowest peak and has fewer than 50% the number of pixels.
    10% and 50% picked empirically to give good results.
    """

    max_peak = np.where(bin_centers == peaks[-1])[0][0]  # Max intensity
    thresh_top = max_peak
    while hist[thresh_top] > hist[max_peak] * 0.1:
        thresh_top += 2  # Upper limit is less sensitive, so step 2 at a time

    min_peak = np.where(bin_centers == peaks[0])[0][0]  # Min intensity
    thresh_bot = min_peak
    while hist[thresh_bot] > hist[min_peak] * 0.5:
        thresh_bot -= 1

    # Convert the histogram bin index to an intensity value
    lower = bin_centers[thresh_bot]
    upper = bin_centers[thresh_top]

    # Limit the amount of stretch to a percentage of the total dynamic range 
    #   in the case that all three main surface types are not represented (fewer
    #   than 3 peaks)
    # 8 bit vs 11 bit (WorldView)
    # 256   or 2048
    # While WV images are 11bit, white ice tends to be ~600-800 intensity
    # Provide a floor and ceiling to the amount of stretch allowed
    if len(peaks) < 3:
        if image_type == 'pan' or image_type == 'wv02_ms':
            max_bit = 2047
            upper_limit = 0.25
        else:
            max_bit = 255
            upper_limit = 0.6
        min_range = int(max_bit * .08)
        if lower > min_range:
            lower = min_range
        # If there are at least 2 peaks we don't need an upper limit, as the upper
        #   limit is only to prevent open water only images from being stretched.
        if len(peaks) < 2:
            max_range = int(max_bit * upper_limit)
            if upper < max_range:
                upper = max_range

    return lower, upper


def split_band(band, num_x, num_y, size_x, size_y):
    """
    Divides the input band into a list of num_x*num_y subregions (splits), each
        of size defined by size_x and size_y
    """
    # Pad the input band with zeros to fit with the desired number and size of
    #   image splits. 
    padded_band = np.zeros([num_y * size_y, num_x * size_x])
    original_dims = np.shape(band)
    padded_band[0:original_dims[0], 0:original_dims[1]] = band
    band = None

    # Create a list of image splits
    all_splits = []
    for y in range(num_y):
        for x in range(num_x):
            split = padded_band[y * size_y:(y + 1) * size_y, x * size_x:(x + 1) * size_x]
            all_splits.append(split)

    return all_splits


def construct_blocks(image, block_cols, block_rows, pad_dim):
    """
    Creates a list of image blocks based on an input raster and desired block
        size.
    Block size needs to be a multiple of total raster dimensions.
    """
    # Pad the input band with zeros to fit with the desired number and size of
    #   image splits. 
    padded_image = np.zeros(pad_dim)
    original_dim = np.shape(image)
    padded_image[0:original_dim[0], 0:original_dim[1]] = image
    image = None

    num_block_cols = int(pad_dim[1] / block_cols)
    num_block_rows = int(pad_dim[0] / block_rows)

    block_list = []
    # Append a 2d array of the image block to
    pad_amt = 100
    for y in range(num_block_rows):
        for x in range(num_block_cols):
            block_list.append(padded_image[(y * block_rows) : ((y + 1) * block_rows),
                                           (x * block_cols) : ((x + 1) * block_cols)])
    dimensions = [num_block_cols, num_block_rows]
    return block_list, dimensions


def write_to_hdf5(dst_file, image_data, band_num, image_type, image_date,
                  dimensions):
    """
    Writes the given image data to an hdf5 file.
    """
    # If the output file already exists, append the given data to that
    #   file. Otherwise, create a new file and add the attribute headers. 
    if os.path.isfile(dst_file):
        outfile = h5py.File(dst_file, "r+")
    else:
        outfile = h5py.File(dst_file, "w")
        outfile.attrs.create("Image Type", image_type)
        outfile.attrs.create("Image Date", image_date)
        outfile.attrs.create("Block Dimensions", dimensions)
    # Catch collisions with existing datasets. If this dataset already exists,
    #   do nothing.
    try:
        outfile.create_dataset('original_' + str(band_num), data=image_data,
                               dtype='uint8', compression='gzip')
    except RuntimeError:
        pass

    outfile.close()


def save_color_image(image_data, output_name, image_type, block_cols, block_rows):
    """
    Write a rgb color image (as png) of the raw image data to disk.
    """
    holder = []
    # Find the appropriate bands to use for an rgb representation
    if image_type == 'wv02_ms':
        rgb = [5, 3, 2]
    elif image_type == 'srgb':
        rgb = [1, 2, 3]
    else:
        rgb = [1, 1, 1]

    red_band = image_data[rgb[0]]
    green_band = image_data[rgb[1]]
    blue_band = image_data[rgb[2]]

    for i in range(len(red_band)):
        holder.append(utils.create_composite([
            red_band[i], green_band[i], blue_band[i]]))

    colorfullimg = utils.compile_subimages(holder, block_cols, block_rows, 3)
    mimg.imsave(output_name, colorfullimg)
    colorfullimg = None


def rescale_band(band, bottom, top):
    """
    Rescale and image data from range [bottom,top] to uint8 ([0,255])
    """
    # Record pixels that contain no spectral information, indicated by a value of 0
    empty_pixels = np.zeros(np.shape(band), dtype='bool')
    empty_pixels[band == 0] = True

    # Rescale the data to use the full int8 (0,255) pixel value range.
    # Check the band where the values of the matrix are greater than zero so that the
    # percentages ignore empty pixels.
    stretched_band = exposure.rescale_intensity(band, in_range=(bottom, top),
                                                out_range=(1, 255))
    new_band = np.array(stretched_band, dtype=np.uint8)
    # Set the empty pixel areas back to a value of 0. 
    new_band[empty_pixels] = 0

    return new_band


def downsample(band, factor):
    """
    'Downsample' an image by the given factor. Every pixel in the resulting image
        is the result of an average of the NxN kernel centered at that pixel,
        where N is factor.
    """

    band_downsample = block_reduce(band, block_size=(factor, factor), func=np.mean)

    band_copy = np.zeros(np.shape(band))
    for i in range(np.shape(band_downsample)[0]):
        for j in range(np.shape(band_downsample)[1]):
            band_copy[i * factor:(i * factor) + factor, j * factor:j * factor + factor] = band_downsample[i, j]

    return band_copy


def main():
    # Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="directory path for input image")
    parser.add_argument("filename",
                        help="name of image")
    parser.add_argument("image_type", type=str, choices=['srgb', 'wv02_ms', 'pan'],
                        help="image type: 'srgb', 'wv02_ms', 'pan'")
    parser.add_argument("--output_dir", metavar="dir",
                        help="directory path for output images")
    parser.add_argument("-s", "--splits", type=int, default=9, metavar="int",
                        help='''number of splits to perform on the input images.
                        This is rounded to the nearest perfect square''')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display status updates")

    # Moved from Splitter, but not implemented yet:
    # parser.add_argument("--histogram", action="store_true",
    #                    help="display histogram of pixel intensity values before segmentation")

    # Parse Arguments
    args = parser.parse_args()
    input_path = os.path.abspath(args.input_dir)
    image_name = args.filename
    image_type = args.image_type
    if args.output_dir:
        output_path = os.path.abspath(args.output_dir)
    else:
        output_path = None
    number_of_splits = args.splits
    verbose = args.verbose

    # Split Image with Given Arguments
    prepare_image(input_path, image_name, image_type,
                  output_path=output_path,
                  number_of_splits=number_of_splits,
                  verbose=verbose)


if __name__ == "__main__":
    main()
