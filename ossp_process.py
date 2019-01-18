# OSSP Process
# Usage: Fully processes all images in the given directory with the given training data.
# Nicholas Wright

import os
import argparse
import csv
import numpy as np
import preprocess as pp
from segment import segment_image
from classify import classify_image
from lib import utils
import gdal


def main():
    # Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help='''directory path containing date directories of 
                        images to be processed''')
    parser.add_argument("image_type", type=str, choices=["srgb", "wv02_ms", "pan"],
                        help="image type: 'srgb', 'wv02_ms', 'pan'")
    parser.add_argument("training_dataset",
                        help="training data file")
    parser.add_argument("--training_label", type=str, default=None,
                        help="name of training classification list")
    parser.add_argument("-o", "--output_dir", type=str, default="default",
                        help="directory to place output results.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display text information and progress")
    # parser.add_argument("-e", "--extended_output", action="store_true",
    #                     help='''Save additional data:
    #                                 1) classified image (png)
    #                                 2) classified results (csv)
    #                     ''')
    parser.add_argument("-c", "--nostretch", action="store_false",
                        help="Do not apply a histogram stretch image correction to input.")

    # Parse Arguments
    args = parser.parse_args()

    # System filepath that contains the directories or files for batch processing
    user_input = args.input_dir
    if os.path.isdir(user_input):
        src_dir = user_input
        src_file = ''
    elif os.path.isfile(user_input):
        src_dir, src_file = os.path.split(user_input)
    else:
        raise IOError('Invalid input')
    # Image type, choices are 'srgb', 'pan', or 'wv02_ms'
    image_type = args.image_type
    # File with the training data
    tds_file = args.training_dataset
    # Default tds label is the image type
    if args.training_label is None:
        tds_label = image_type
    else:
        tds_label = args.training_label
    # Default output directory
    #   (if not provided this gets set when the tasks are created)
    dst_dir = args.output_dir

    verbose = args.verbose
    # extended_output = args.extended_output
    stretch = args.nostretch

    # For Ames OIB Processing:
    if image_type == 'srgb':
        assess_quality = True
    else:
        assess_quality = False
    # Set a default quality score until this value is calculated
    quality_score = 1.

    # Prepare a list of images to be processed based on the user input
    #   list of task objects based on the files in the input directory.
    #   Each task is an image to process, and has a subtask for each split
    #   of that image. 
    task_list = utils.create_task_list(os.path.join(src_dir, src_file), dst_dir)

    # Load Training Data
    tds = utils.load_tds(tds_file, tds_label)

    for task in task_list:

        # ASP: Restrict processing to the frame range
        # try:
        #     frameNum = getFrameNumberFromFilename(file)
        # except Exception, e:
        #     continue
        # if (frameNum < args.min_frame) or (frameNum > args.max_frame):
        #     continue

        # Skip this task if it is already marked as complete
        if task.is_complete():
            continue

        # Make the output directory if it doesnt already exist
        if not os.path.isdir(task.get_dst_dir()):
            os.makedirs(task.get_dst_dir())

        # Open the image dataset with gdal
        full_image_name = os.path.join(src_dir, task.get_id())
        if os.path.isfile(full_image_name):
            if verbose:
                print("Loading image...")
            src_ds = gdal.Open(full_image_name, gdal.GA_ReadOnly)
        else:
            print("File not found: {}".format(full_image_name))
            continue

        # Read metadata to get image
        metadata = src_ds.GetMetadata()
        image_date = pp.parse_metadata(metadata, image_type)

        # Set necessary parameters for reading image 1 block at a time
        x_dim = src_ds.RasterXSize
        y_dim = src_ds.RasterYSize
        desired_block_size = 6400

        # Analyze input image histogram (if applying correction)
        if stretch:
            lower, upper = pp.histogram_threshold(src_ds, image_type)
        elif image_type == 'wv02_ms' or image_type == 'pan':
            lower = 1
            upper = 2047
        # Can assume srgb images are already 8bit
        else:
            lower = 1
            upper = 255

        # Create a blank output image dataset
        # Save the classified image output as a geotiff
        fileformat = "GTiff"
        image_name_noext = os.path.splitext(task.get_id())[0]
        dst_filename = os.path.join(task.get_dst_dir(), image_name_noext + '_classified.tif')
        driver = gdal.GetDriverByName(fileformat)
        dst_ds = driver.Create(dst_filename, xsize=x_dim, ysize=y_dim,
                               bands=1, eType=gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=LZW"])

        # Transfer the metadata from input image
        # dst_ds.SetMetadata(src_ds.GetMetadata())
        # Transfer the input projection
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())  # sets same geotransform as input
        dst_ds.SetProjection(src_ds.GetProjection())  # sets same projection as input

        # Set an empty value for the pixel counter
        pixel_counts = [0, 0, 0, 0, 0]

        # Find the appropriate image block read size
        block_size_x, block_size_y = utils.find_blocksize(x_dim, y_dim, desired_block_size)
        if verbose:
            print("block size: [{},{}]".format(block_size_x, block_size_y))
        # Convert the block size into a list of the top (y) left (x) coordinate of each block
        #   and iterate over both lists to process each block
        y_blocks = range(0, y_dim, block_size_y)
        x_blocks = range(0, x_dim, block_size_x)

        # Display a progress bar
        if verbose:
            try:
                from tqdm import tqdm
            except ImportError:
                print "Install tqdm to display progress bar."
                verbose = False
            else:
                pbar = tqdm(total=len(y_blocks)*len(x_blocks)*2, unit='block')

        # Iterate over the image blocks
        for y in y_blocks:
            # Check that this block will lie within the image dimensions
            read_size_y = check_read_size(y, block_size_y, y_dim)

            for x in x_blocks:
                # Check that this block will lie within the image dimensions
                read_size_x = check_read_size(x, block_size_x, x_dim)

                # Load block data with gdal (offset and block size)
                image_data = src_ds.ReadAsArray(x, y, read_size_x, read_size_y)

                # Restructure raster for panchromatic images:
                if image_data.ndim == 2:
                    image_data = np.reshape(image_data, (1, read_size_y, read_size_x))

                # Calcualate the quality score on an arbitrary band
                if assess_quality:
                    quality_score = pp.calc_q_score(image_data[0])

                # Apply correction to block based on earlier histogram analysis (if applying correction)
                # Converts image to 8 bit by rescaling lower -> 1 and upper -> 255
                image_data = pp.rescale_band(image_data, lower, upper)

                # Segment image
                segmented_blocks = segment_image(image_data, image_type=image_type)

                # Update the progress bar
                if verbose: pbar.update()

                # Classify image
                classified_block = classify_image(image_data, segmented_blocks,
                                                  tds, [image_type, image_date])

                # Add the pixel counts from this classified split to the
                #   running total.
                pixel_counts_block = utils.count_features(classified_block)
                for i in range(len(pixel_counts)):
                    pixel_counts[i] += pixel_counts_block[i]

                # Write information to output
                dst_ds.GetRasterBand(1).WriteArray(classified_block, xoff=x, yoff=y)
                dst_ds.FlushCache()
                # dst_ds = None
                # quit()

                # Update the progress bar
                if verbose: pbar.update()

        # Close dataset and write to disk
        dst_ds = None
        src_ds = None

        # Write extra data (total pixel counts and quality score to the database (or csv)
        output_csv = os.path.join(task.get_dst_dir(), image_name_noext + '_md.csv')
        with open(output_csv, "wb") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Quality Score", "White Ice", "Gray Ice", "Melt Ponds", "Open Water"])
            writer.writerow([quality_score, pixel_counts[0], pixel_counts[1], pixel_counts[2], pixel_counts[3]])

        # # Save color image for viewing
        # if extended_output:
        #     utils.save_color(classified_image,
        #                      os.path.join(dst_dir, image_name + '.png'))

        # Close the progress bar
        if verbose:
            pbar.close()
            print "Finished Processing."


def check_read_size(y, block_size_y, y_dim):
    if y + block_size_y < y_dim:
        return block_size_y
    else:
        return y_dim - y


if __name__ == "__main__":
    main()
