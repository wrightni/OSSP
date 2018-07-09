# title: Watershed Transform
# author: Nick Wright
# adapted from: Justin Chen, Arnold Song

import argparse
import time
import numpy as np
import os
import h5py
from multiprocessing import Process, Queue

from skimage import filters, morphology, feature, exposure
from skimage.future import graph
from scipy import ndimage

from lib import utils
from lib import debug_tools

# tqdm for progress bar


def segment_image(input_data, image_type=False, test_check=False, threads=2,
                  write_results=False, dst_file=False, verbose=False):
    '''
    Wrapper function that handles all of the processing to create watersheds
    '''

    #### Check the input format and read appropriate values
    try:
        data_file = h5py.File(input_data,'r')
    except TypeError:
        # Input_data is not a string (so it is image data?)
        im_block_dict = input_data     # Does this need to be checked first?
        input_data = None
        pass
    except IOError:
        # Input_data is string, but not a valid hdf5 file
        raise
    else:
        # We opened the file to test, but aren't reading it here.
        data_file.close()
        # Destination file for writing watersheds is the same as the input file
        dst_file = input_data
        im_block_dict, image_type = load_from_disk(input_data, verbose)


    #### Check for empty image data, if found, save placeholder watershed
    # if im_block_dict == None and np.sum(full_band) == 0:
    #     if verbose: print "Saving output files..."
    #     placeholder_data = {1:full_band}
    #     placeholder_watershed = np.zeros(np.shape(full_band))
    #     write_to_hdf5(placeholder_data, placeholder_watershed, dst_filename,
    #                     image_type, image_data, [num_x_subimages,num_y_subimages])
    #     if verbose: print "Done."
    #     return None
    # elif im_block_dict == None:
    #     return  None #why?
 
    #### Define Amplification and Threshold
    # These values are dependent on the type of imagery being processed, and are
    #   mostly empirically derived.
    # band_list contains the three bands to be used for segmentation
    if image_type == 'pan':
        sobel_threshold = 0.1
        amplification_factor = 2.
        band_list = [1,1,1]
    elif image_type == 'wv02_ms':
        sobel_threshold = 0.0
        amplification_factor = 3
        band_list = [5,3,2]
    elif image_type == 'srgb':
        sobel_threshold = 0.1
        amplification_factor = 3.
        band_list = [3,2,1]


    #### Segment each image block
    # segmnt_block_queue is a queue that stores the result of the watershed
    #   segmentation, where each element is one image block. 
    segmnt_block_queue = Queue()
    num_blocks = len(im_block_dict[1])
    block_queue = construct_block_queue(im_block_dict, band_list, num_blocks)
    # im_block_dict = None

    
    # Define the number of threads to create
    NUMBER_OF_PROCESSES = threads
    block_procs = [Process(target=process_block_helper, 
                           args=(block_queue, segmnt_block_queue, 
                                 sobel_threshold, amplification_factor))
                   for _ in range(NUMBER_OF_PROCESSES)]
    
    # Start the worker processes. 
    for proc in block_procs:
        # Add a stop command to the end of the queue for each of the 
        #   processes started. This will signal for the process to stop. 
        block_queue.put('STOP')
        # Start the process
        proc.start()

    # Display a progress bar
    if verbose:
        try:
            from tqdm import tqdm
        except ImportError:
            print "Install tqdm to display progress bar."
            verbose = False
        else:
            pbar = tqdm(total=num_blocks, unit='block')

    # Each process adds the output values to segmnt_block_queue when it 
    #   finishes a row. Adds 'None' when there are no more rows left 
    #   in the queue. 
    # This loop continues as long as all of the processes have not finished
    #   (i.e. fewer than NUMBER_OF_PROCESSES have returned None). When a row is 
    #   added to the output list, the tqdm progress bar updates.

    # Initialize the output dataset as an empty list of length = input dataset
    #   This needs to be initialized since blocks will be added non-sequentially
    segmnt_block_list = [None for _ in range(num_blocks)]
    finished_threads = 0
    while finished_threads < NUMBER_OF_PROCESSES:
        if not segmnt_block_queue.empty():
            val = segmnt_block_queue.get()
            if val == None:
                finished_threads += 1
            else:
                block_num = val[0]
                segmnt_data = val[1]
                segmnt_block_list[block_num] = segmnt_data
                if verbose: pbar.update()

    # Close the progress bar
    if verbose: 
        pbar.close()
        print "Finished Processing. Closing threads..."

    # Join all of the processes back together
    for proc in block_procs:
        proc.join()

    while test_check:
        test_check = check_results(im_block_dict,segmnt_block_list)
    # print np.shape(segmnt_block_list)
    if write_results:
        write_to_hdf5(segmnt_block_list, dst_file)
        return None, None
    else:
        return im_block_dict, segmnt_block_list


def construct_block_queue(im_block_dict,band_list,size):
    '''
    Constructs a multiprocessing queue from a list of image blocks, where
        each item in the queue is a single block and its list index. 
    '''
    # Create a multiprocessing Queue
    block_queue = Queue()
    # Add each block to the queue with the index (to track block location).
    for x in range(size):
        block_queue.put([x,[im_block_dict[band_list[0]][x],
                            im_block_dict[band_list[1]][x],
                            im_block_dict[band_list[2]][x]]])
    return block_queue


def process_block_helper(im_block_queue, segmented_blocks, 
                         s_threshold, amp_factor):
    '''
    Function run by each thread. Acquires the next block from im_block_queue and
        gives it to process_block(). Continues until there are no more 
        blocks left in the queue. 
    '''
    # Read the next item in the queue until the 'STOP' command has been
    #  reached. 
    for block_num, block in iter(im_block_queue.get, 'STOP'):
        # Process the next block of data
        result = watershed_transformation(block, s_threshold, amp_factor)
        # Write the results to the output queue
        segmented_blocks.put([block_num,result])
    # Signal that this process has finished its task
    segmented_blocks.put(None)


def watershed_transformation(image_data, sobel_threshold, amplification_factor):
    '''
    Runs a watershed transform on the main dataset
        1. Create a gradient image using the sobel algorithm
        2. Adjust the gradient image based on given threshold and amplification.
        3. Find the local minimum gradient values and place a marker
        4. Construct watersheds on top of the gradient image starting at the 
            markers.
        5. Recombine neighboring image segments using a region adjacency graph.
    '''
    # If this block has no data, return a placeholder watershed.
    if np.amax(image_data[0]) <= 1:
        return np.zeros(np.shape(image_data))

    # Create a gradient image using a sobel filter
    sobel_image = filters.sobel(image_data[0])

    # Adjust the sobel image based on the given threshold and amp factor.
    upper_threshold = np.amax(sobel_image)/amplification_factor
    if upper_threshold < 0.20:
        upper_threshold = 0.20
    sobel_image = exposure.rescale_intensity(sobel_image, 
                                             in_range=(0,upper_threshold), 
                                             out_range=(0,1))

    # Prevent the watersheds from 'leaking' along the sides of the image
    sobel_image[:,0] = 1
    sobel_image[:,-1] = 1
    sobel_image[0,:] = 1
    sobel_image[-1,:] = 1

    # Set all values in the sobel image that are lower than the 
    #   given threshold to zero.
    sobel_image[sobel_image<=sobel_threshold]=0

    #sobel_copy = np.copy(sobel_image)

    # Find local minimum values in the sobel image by inverting
    #   sobel_image and finding the local maximum values
    inv_sobel = 1-sobel_image
    local_min = feature.peak_local_max(inv_sobel, min_distance=2, 
                                       indices=False, num_peaks_per_label=1)
    markers = ndimage.label(local_min)[0]

    # Build a watershed from the markers on top of the edge image
    im_watersheds = morphology.watershed(sobel_image,markers)
    im_watersheds = np.array(im_watersheds,dtype='uint16')
    # Clear gradient image data
    sobel_image = None

    # Find the locations that contain no spectral data
    empty_pixels = np.zeros(np.shape(image_data[0]),dtype='bool')
    empty_pixels[image_data[0] == 0] = True
    # Set all values outside of the image area (empty pixels, usually caused by
    #   orthorectification) to one value, at the end of the watershed list.
    im_watersheds[empty_pixels] = np.amax(im_watersheds)+1

    # Recombine segments that are adjacent and similar to each other. 
    #   Created a region adjacency graph. Create_composite() takes a single list 
    #   of bands
    color_im = utils.create_composite([image_data[0], image_data[1], image_data[2]])

    # Clear image data
    image_data = None
    # Create the region adjacency graph based on the color image
    try:
        im_graph = graph.rag_mean_color(color_im,im_watersheds)
    except KeyError:
        pass
    else:
        # Clear color image data
        color_im = None
        # Combine segments that are adjacent and whose pixel intensity 
        #   difference is less than 10. 
        im_watersheds = graph.cut_threshold(im_watersheds,im_graph,5.0)

    return im_watersheds

        
def load_from_disk(hdf5_file, verbose):
    '''
    Reads hdf5 file of an image split
    Returns:
        Dictionary containing image data
            keys: band number (1-8 possible), as int
            value: [[block][row][column]]
        Image type variable
    
    Note: All bands are read from the file, though only bands 1,5,3,2 are used
        in the segmentation process. The rest are passed to the classification
        script.
    '''
    if verbose: 
        start_time = time.clock()
        print "Reading file..."

    # Load all band datasets from the given file. 
    with h5py.File(hdf5_file, 'r') as f: 
        # Read image type from file header
        image_type = f.attrs.get("Image Type")
        all_band_blocks = {}  # {band_id: [block][row][column]}
        # The inputfile has 1 dataset for every band (see preprocess.py)
        
        # Don't count a watershed dataset as a band
        datasets = f.keys()
        num_bands = len(datasets)
        if 'watershed' in datasets:
            num_bands -= 1

        for b in range(1,num_bands+1):
            dataset_name = 'original_' + str(b)
            band = f[dataset_name][:]
            all_band_blocks[b] = band
            # debug_tools.display_histogram(band[40])

    if verbose: 
        elapsed_time = time.clock() - start_time    
        print "Done. "
        print "Time elapsed: {0}".format(elapsed_time)

    return all_band_blocks, image_type


def write_to_hdf5(watershed_data, dst_file):
    '''
    Saves the watershed data to the input data file. 
    '''
    with h5py.File(dst_file, "r+") as outfile:
        try:
            outfile.create_dataset('watershed', data=watershed_data,
                                    compression='gzip',compression_opts=3)
        except RuntimeError:
            pass
 

def check_results(optic_subimage_dict, ws_subimage_list):

    choice = raw_input("Display subimage/wsimage pair? (y/n): ")
    if choice == 'y':
        selection = int(raw_input("Choose image (0," +str(len(ws_subimage_list)-1) + "): "))
        if selection >= 0 and selection < len(ws_subimage_list)-1:
            debug_tools.display_watershed(optic_subimage_dict,
                                          ws_subimage_list,
                                          block=selection)
            # subimage_list[selection].display_watershed()
        else:
            print "Invalid subimage index."
        test_check = True
    else:
        test_check = False
        save_flag = raw_input("Save results? (y/n): ")
        if save_flag == 'n':
            quit()
    return test_check


#get_distance and get_markers are methods used in the watershed transformation to create
#   the seed points based on the binary Otsu image. 
def get_distance(image):
    distance_image = ndimage.distance_transform_edt(image)
    shape = np.shape(distance_image)
    extenshape = (shape[0]+20,shape[1]+20)
    extendist = np.zeros(extenshape)
    extendist[10:shape[0]+10,10:shape[1]+10]=distance_image

    return extendist, shape

def get_markers(extendist, shape):
    extenlocal_max = feature.peak_local_max(extendist,min_distance=3, 
                                            threshold_rel=0., indices=False)
    local_max = extenlocal_max[10:shape[0]+10,10:shape[1]+10]
    markers = ndimage.label(local_max)[0]

    return markers


def main():
    
    #### Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", 
                        help="directory containing a folder of image splits in .h5 format")
    parser.add_argument("filename",
                        help="name of image split")
    parser.add_argument("-p", "--parallel", metavar='int', type=int, default=1,
                        help='''number of processing threads to create.''')
    parser.add_argument("-w", "--write", action="store_true",
                        help="write the segmented image data to the input file")
    parser.add_argument("-t", "--test", action="store_true",
                        help="inspect results before saving")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display text information and progress")

    #### Parse Arguments
    args = parser.parse_args()
    src_path = os.path.abspath(args.input_dir)
    image_name = args.filename
    parallel = args.parallel
    write_results = args.write
    test_check = args.test
    verbose = args.verbose

    # Combine image name and filepath
    image_path = os.path.join(src_path, image_name)

    segment_image(image_path, test_check=test_check, threads=parallel,
                  write_results=write_results, dst_file=src_path, verbose=verbose)

if __name__ == "__main__":
    main()
