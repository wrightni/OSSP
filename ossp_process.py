# OSSP Process
# Usage: Fully processes all images in the given directory with the given training data.
# Nicholas Wright

import os
import time
import argparse
import csv
import numpy as np
from multiprocessing import Process, RLock, Queue
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
    parser.add_argument("-c", "--stretch",
                        type=str,
                        choices=["hist", "pansh", "none"],
                        default='hist',
                        help='''Apply image correction/stretch to input: \n
                               hist: Histogram stretch \n
                               pansh: Orthorectify / Pansharpen for MS WV images \n
                               none: No correction''')
    parser.add_argument("--pgc_script", type=str, default=None,
                        help="Path for the pansharpening script if needed")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Number of subprocesses to start")

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
    threads = args.threads
    verbose = args.verbose
    stretch = args.stretch


    # Use the given pansh script path, otherwise search for the correct folder
    #   in the same directory as this script.
    if args.pgc_script:
        pansh_script_path = args.pgc_script
    else:
        current_path = os.path.dirname(os.path.realpath(__file__))
        pansh_script_path = os.path.join(os.path.split(current_path)[0], 'imagery_utils')

    # For Ames OIB Processing:
    # White balance flag (To add as user option in future, presently only used on oib imagery)
    if image_type == 'srgb':
        assess_quality = True
        white_balance = True
    else:
        assess_quality = False
        white_balance = False
    # Set a default quality score until this value is calculated
    quality_score = 1.

    # Prepare a list of images to be processed based on the user input
    #   list of task objects based on the files in the input directory.
    #   Each task is an image to process, and has a subtask for each split
    #   of that image. 
    task_list = utils.create_task_list(os.path.join(src_dir, src_file), dst_dir)

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

        # Run Ortho/Pan scripts if necessary
        if stretch == 'pansh':
            if verbose:
                print("Orthorectifying and Pansharpening image...")

            full_image_name = os.path.join(task.get_src_dir(), task.get_id())
            pansh_filepath = pp.run_pgc_pansharpen(pansh_script_path,
                                                   full_image_name,
                                                   task.get_dst_dir())

            # Set the image name/dir to the pan output name/dir
            task.set_src_dir(task.get_dst_dir())
            task.change_id(pansh_filepath)

        # Open the image dataset with gdal
        full_image_name = os.path.join(task.get_src_dir(), task.get_id())
        if os.path.isfile(full_image_name):
            if verbose:
                print("Loading image {}...".format(task.get_id()))
            src_ds = gdal.Open(full_image_name, gdal.GA_ReadOnly)
        else:
            print("File not found: {}".format(full_image_name))
            continue

        # Read metadata to get image date and keep only the metadata we need
        metadata = src_ds.GetMetadata()
        image_date = pp.parse_metadata(metadata, image_type)
        metadata = [image_type, image_date]

        # For processing icebridge imagery:
        if image_type == 'srgb':
            if image_date <= 150:
                tds_label = 'spring'
                white_balance = True
            else:
                tds_label = 'summer'

        # Load Training Data
        tds = utils.load_tds(tds_file, tds_label, image_type)
        # tds = utils.load_tds(tds_file, 'srgb', image_type)

        if verbose:
            print("Size of training set: {}".format(len(tds[1])))

        # Set necessary parameters for reading image 1 block at a time
        x_dim = src_ds.RasterXSize
        y_dim = src_ds.RasterYSize
        desired_block_size = 6400

        src_dtype = gdal.GetDataTypeSize(src_ds.GetRasterBand(1).DataType)
        # Analyze input image histogram (if applying correction)
        if stretch == 'hist':
            stretch_params = pp.histogram_threshold(src_ds, src_dtype)
        else:  # stretch == 'none':
            # WV Images are actually 11bit stored in 16bit files
            if src_dtype > 12:
                src_dtype = 11
            stretch_params = [1, 2**src_dtype - 1,
                              [2 ** src_dtype - 1 for _ in range(src_ds.RasterCount)],
                              [1 for _ in range(src_ds.RasterCount)]]

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
        # Transfer the input projection and geotransform if they are different than the default
        if src_ds.GetGeoTransform() != (0, 1, 0, 0, 0, 1):
            dst_ds.SetGeoTransform(src_ds.GetGeoTransform())  # sets same geotransform as input
        if src_ds.GetProjection() != '':
            dst_ds.SetProjection(src_ds.GetProjection())  # sets same projection as input

        # Find the appropriate image block read size
        block_size_x, block_size_y = utils.find_blocksize(x_dim, y_dim, desired_block_size)
        if verbose:
            print("block size: [{},{}]".format(block_size_x, block_size_y))

        # close the source dataset so that it can be loaded by each thread seperately
        src_ds = None
        lock = RLock()
        block_queue, qsize = construct_block_queue(block_size_x, block_size_y, x_dim, y_dim)
        dst_queue = Queue()

        # Display a progress bar
        if verbose:
            try:
                from tqdm import tqdm
            except ImportError:
                print("Install tqdm to display progress bar.")
                verbose = False
            else:
                pbar = tqdm(total=qsize, unit='block')

        # Set an empty value for the pixel counter
        pixel_counts = [0, 0, 0, 0, 0]

        NUMBER_OF_PROCESSES = threads
        block_procs = [Process(target=process_block_queue,
                               args=(lock, block_queue, dst_queue, full_image_name,
                                     assess_quality, stretch_params, white_balance, tds, metadata))
                       for _ in range(NUMBER_OF_PROCESSES)]

        for proc in block_procs:
            # Add a stop command to the end of the queue for each of the
            #   processes started. This will signal for the process to stop.
            block_queue.put('STOP')
            # Start the process
            proc.start()

        # Collect data from processes as they complete tasks
        finished_threads = 0
        while finished_threads < NUMBER_OF_PROCESSES:

            if not dst_queue.empty():
                val = dst_queue.get()
                if val is None:
                    finished_threads += 1
                else:
                    # Keep only the lowest quality score found
                    quality_score_block = val[0]
                    if quality_score_block < quality_score:
                        quality_score = quality_score_block
                    # Add the pixel counts to the master list
                    pixel_counts_block = val[1]
                    for i in range(len(pixel_counts)):
                        pixel_counts[i] += pixel_counts_block[i]
                    # Write image data to output dataset
                    x = val[2]
                    y = val[3]
                    classified_block = val[4]
                    dst_ds.GetRasterBand(1).WriteArray(classified_block, xoff=x, yoff=y)
                    dst_ds.FlushCache()
                    # Update the progress bar
                    if verbose: pbar.update()
            # Give the other threads some time to finish their tasks.
            else:
                time.sleep(10)

        # Update the progress bar
        if verbose: pbar.update()

        # Join all of the processes back together
        for proc in block_procs:
            proc.join()

        # Close dataset and write to disk
        dst_ds = None

        # Write extra data (total pixel counts and quality score to the database (or csv)
        output_csv = os.path.join(task.get_dst_dir(), image_name_noext + '_md.csv')
        with open(output_csv, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Quality Score", "White Ice", "Gray Ice", "Melt Ponds", "Open Water", "Shadow"])
            writer.writerow([quality_score, pixel_counts[0], pixel_counts[1], pixel_counts[2],
                             pixel_counts[3], pixel_counts[4]])

        # Close the progress bar
        if verbose:
            pbar.close()
            print("Finished Processing.")


def construct_block_queue(block_size_x, block_size_y, x_dim, y_dim):
    # Convert the block size into a list of the top (y) left (x) coordinate of each block
    #   and iterate over both lists to process each block
    y_blocks = range(0, y_dim, block_size_y)
    x_blocks = range(0, x_dim, block_size_x)
    qsize = 0
    # Construct a queue of block coordinates
    block_queue = Queue()
    for y in y_blocks:
        for x in x_blocks:
            # Check that this block will lie within the image dimensions
            read_size_y = check_read_size(y, block_size_y, y_dim)
            read_size_x = check_read_size(x, block_size_x, x_dim)
            # Store variables needed to read each block from source dataset in queue
            block_queue.put((x, y, read_size_x, read_size_y))
            qsize += 1

    return block_queue, qsize


def process_block_queue(lock, block_queue, dst_queue, full_image_name,
                        assess_quality, stretch_params, white_balance, tds, im_metadata):
    '''
    Function run by each process. Will process blocks placed in the block_queue until the 'STOP' command is reached.
    '''
    # Parse input arguments
    lower, upper, wb_reference, bp_reference = stretch_params
    wb_reference = np.array(wb_reference, dtype=np.float)
    bp_reference = np.array(bp_reference, dtype=np.float)
    image_type = im_metadata[0]

    for block_indices in iter(block_queue.get, 'STOP'):

        x, y, read_size_x, read_size_y = block_indices
        # Load block data with gdal (offset and block size)
        lock.acquire()
        src_ds = gdal.Open(full_image_name, gdal.GA_ReadOnly)
        image_data = src_ds.ReadAsArray(x, y, read_size_x, read_size_y)
        src_ds = None
        lock.release()

        # Restructure raster for panchromatic images:
        if image_data.ndim == 2:
            image_data = np.reshape(image_data, (1, read_size_y, read_size_x))

        # Calculate the quality score on an arbitrary band
        if assess_quality:
            quality_score = pp.calc_q_score(image_data[0])
        else:
            quality_score = 1.
        # Apply correction to block based on earlier histogram analysis (if applying correction)
        # Converts image to 8 bit by rescaling lower -> 1 and upper -> 255
        image_data = pp.rescale_band(image_data, lower, upper)
        if white_balance:
            # Applies a white balance correction
            image_data = pp.white_balance(image_data, wb_reference, np.amax(wb_reference))

        # Segment image
        segmented_blocks = segment_image(image_data, image_type=image_type)

        # Classify image
        classified_block = classify_image(image_data, segmented_blocks,
                                          tds, im_metadata, wb_reference, bp_reference)

        # Add the pixel counts from this classified split to the
        #   running total.
        pixel_counts_block = utils.count_features(classified_block)

        # Pass the data back to the main thread for writing
        dst_queue.put((quality_score, pixel_counts_block, x, y, classified_block))

    dst_queue.put(None)


def check_read_size(y, block_size_y, y_dim):
    if y + block_size_y < y_dim:
        return block_size_y
    else:
        return y_dim - y


if __name__ == "__main__":
    main()
