#title: Random Forest Classifier
#author: Nick Wright

import argparse
from multiprocessing import Process, Queue

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk

from lib import utils, feature_calculations

# tqdm for progress bar


def classify_image(input_image, watershed_data, training_dataset, meta_data,
            threads=1, quality_control=False, verbose=False):
    '''
    Run a random forest classification. 
    Input: 
        input_image: preprocessed image data (preprocess.py)
        watershed_image: Image objects created with the segmentation 
            algorithm. (segment.py)
        training_dataset: Tuple of training data in the form:
            (label_vector, attribute_matrix)
        meta_data: [im_type, im_date]
    Returns:
        Raster of classified data. 
    '''

    #### Prepare Data and Variables
    num_blocks = len(input_image[1])
    num_bands = len(input_image.keys())
    image_type = meta_data[0]
    image_date = meta_data[1]

    ## Restructure the input data.
    # We are creating a single list where each element of the list is one
    #   block (old: subimage) of the image and is a stack of all bands.  
    image_data = []    # [block:row:column:band]
    for blk in range(num_blocks):
        image_data.append(utils.create_composite(
                [input_image[b][blk] for b in range(1,num_bands+1)]))
    input_image = None

    ## Parse training_dataset input
    label_vector = training_dataset[0]
    training_feature_matrix = training_dataset[1]

    #Method for assessing the quality of the training dataset. 
    if quality_control == True:
        test_training(label_vector, training_feature_matrix)
        aa = raw_input("Continue? ")
        if aa == 'n':
            quit()

    # # If there is no information in this image file, save a dummy classified image and exit
    # # This can often happen depending on the original image dimensions and the amount it was split
    # if np.sum(band_1) == 0:
    #     classified_image_path = os.path.join(output_filepath, output_filename + '_classified_image.png')
    #     outfile = h5py.File(os.path.join(output_filepath, output_filename + '_classified.h5'),'w')
        
    #     if im_type == 'wv02_ms':
    #             empty_bands = np.zeros(np.shape(band_1)[0],np.shape(band_1)[1],8)
    #                     empty_image = utils.compile_subimages(empty_bands, num_x_subimages, num_y_subimages, 8)
    #             elif im_type == 'srgb':
    #                     empty_bands = np.zeros(np.shape(band_1)[0],np.shape(band_1)[1],3)
    #                     empty_image = utils.compile_subimages(empty_bands, num_x_subimages, num_y_subimages, 3)
    #             elif im_type == 'pan':
    #                     empty_image = np.zeros(np.shape(band_1))
        
    #     outfile.create_dataset('classified', data=empty_image,compression='gzip',compression_opts=9)
    #     outfile.create_dataset('original', data=empty_image,compression='gzip',compression_opts=9)
    #     outfile.close()
    #     # return a 1x5 array with values of one for the pixel counts
    #     return output_filename, np.ones(5)

    #### Construct the random forest decision tree using the training data set
    rfc = RandomForestClassifier()
    rfc.fit(training_feature_matrix, label_vector)

    #### Classify each image block
    # Define multiprocessing-safe queues containing data to process
    clsf_block_queue = Queue()
    num_blocks = len(watershed_data)
    im_block_queue = construct_block_queue(image_data, watershed_data, num_blocks)

    # Define the number of threads to create
    NUMBER_OF_PROCESSES = threads
    block_procs = [Process(target=process_block_helper,
                           args=(im_block_queue, clsf_block_queue, image_type, 
                                 image_date, rfc))
                   for _ in range(NUMBER_OF_PROCESSES)]

    # Start the worker processes. 
    for proc in block_procs:
        # Add a stop command to the end of the queue for each of the 
        #   processes started. This will signal for the process to stop. 
        im_block_queue.put('STOP')
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

    # Each process adds the classification results to clsf_block_queue, when it 
    #   finishes a row. Adds 'None' when there are not more rows left 
    #   in the queue. 
    # This loop continues as long as all of the processes have not finished
    #   (i.e. fewer than NUMBER_OF_PROCESSES have returned None). When a row is 
    #   added to the queue, the tqdm progress bar updates.

    # Initialize the output dataset as an empty list of length = input dataset
    #   This needs to be initialized since blocks will be added non-sequentially
    clsf_block_list = [None for _ in range(num_blocks)]
    finished_threads = 0
    while finished_threads < NUMBER_OF_PROCESSES:
        if not clsf_block_queue.empty():
            val = clsf_block_queue.get()
            if val == None:
                finished_threads += 1
            else:
                block_num = val[0]
                segmnt_data = val[1]
                clsf_block_list[block_num] = segmnt_data
                if verbose: pbar.update()

    # Close the progress bar
    if verbose: 
        pbar.close()
        print "Finished Processing. Closing threads..."

    # Join all of the processes back together
    for proc in block_procs:
        proc.join()

    return clsf_block_list


    # # Lite version: Save only the classified output, and do not save the original image data
    # compiled_classified = utils.compile_subimages(classified_image, num_x_subimages, num_y_subimages, 1)
    #
    # if verbose: print "Saving..."
    #
    # with h5py.File(os.path.join(output_filepath, output_filename + '_classified.h5'),'w') as outfile:
    #     outfile.create_dataset('classified', data=compiled_classified,compression='gzip',compression_opts=9)
    #
    # #### Count the number of pixels that were in each classification category.
    # sum_snow, sum_gray_ice, sum_melt_ponds, sum_open_water, sum_shadow = utils.count_features(compiled_classified)
    # pixel_counts = [sum_snow, sum_gray_ice, sum_melt_ponds, sum_open_water, sum_shadow]
    #
    # # Clear the image datasets from memory
    # compiled_classified = None
    # input_image = None
    # watershed_image = None
    #
    # cur_image = None
    # cur_ws = None
    # entropy_image = None
    #
    # if verbose: print "Done."
    #
    # return output_filename, pixel_counts


def construct_block_queue(image_block_list, watershed_block_list, size):
    '''
    Constructs a multiprocessing queue that contains all of the data needed to
        classify a single block of image data.
    Each item in the queue contains the original image data and the segmented
        image. 
    '''
    # Create a multiprocessing Queue
    block_queue = Queue()
    # Add each block to the queue with the index (to track block location).
    for x in range(size):
        block_queue.put([x,image_block_list[x], watershed_block_list[x]])
    return block_queue

def process_block_helper(im_block_queue, clsf_block_queue, image_type, 
                         image_date, rfc):
    '''
    Function run by each thread. Acquires the next block from the queue and
        gives it to process_block(). Continues until there are no more 
        blocks left in the queue. 
    '''
    # Read the next item in the queue until the 'STOP' command has been
    #  reached. 
    for block_num, im_block, ws_block in iter(im_block_queue.get, 'STOP'):
        # Process the next block of data
        result = classify_block(im_block, ws_block, image_type, image_date, rfc)
        # Write the results to the output queue
        clsf_block_queue.put([block_num,result])

        # debug_tools.display_image(im_block,result,2)
        # time.sleep(10)
    # Signal that this process has finished its task
    clsf_block_queue.put(None)


def classify_block(image_block, watershed_block, image_type, image_date, rfc):

    clsf_block = []

    ## If the block contains no data, set the classification values to 0 
    if np.amax(image_block) < 2:
        clsf_block = np.zeros(np.shape(image_block)[0:2])
        return clsf_block

    ## We need the object labels to start at 0. This shifts the entire 
    #   label image down so that the first label is 0, if it isn't already. 
    if np.amin(watershed_block) > 0:
        watershed_block -= np.amin(watershed_block)

    ## Calculate the features of each segment within the block. This 
    #   calculation is unique for each image type. 
    if image_type == 'wv02_ms':
        input_feature_matrix = feature_calculations.analyze_ms_image(
                                image_block, watershed_block)
    elif image_type == 'srgb':
        entropy_image = entropy(image_block[:,:,0], disk(4))
        input_feature_matrix = feature_calculations.analyze_srgb_image(
                                image_block, watershed_block, entropy_image)
    elif image_type == 'pan':
        entropy_image = entropy(image_block[:,:,0], disk(4))
        input_feature_matrix = feature_calculations.analyze_pan_image(
                                image_block, watershed_block, 
                                entropy_image, image_date)

    input_feature_matrix = np.array(input_feature_matrix)

    ## Predict the classification of each segment
    ws_predictions = rfc.predict(input_feature_matrix)

    # Create the classified image by replacing watershed id's with 
    #   classification values.
    # If there is more than one band, we have to select one (using 2 for 
    #   no particular reason).
    # if image_type == 'pan':
    #     clsf_block = create_clsf_raster(ws_predictions, watershed_block, 
    #                                     image_block)
    # else:
    clsf_block = create_clsf_raster(ws_predictions, watershed_block, 
                                    image_block[:,:,0])
    return clsf_block


def create_clsf_raster(prediction, watershed_block, image_block):
    '''
    Transfer classified results from a list of segment:classification pairs
        to a raster where pixel values are the classification result. 
    '''
    # Create a blank image that we will assign values based on the prediction for each
    #   watershed. 
    clsf_block = np.zeros(np.shape(image_block),dtype=np.uint8)
    
    # Check to see if the whole block is one segment
    if np.amax(watershed_block) == 1:
        clsf_block = clsf_block + prediction[0]
        clsf_block[image_block == 0] = 0
        return clsf_block

    # Watershed indexes start at 0, so we have to add 1 to get the number. 
    num_watersheds = int(np.amax(watershed_block)+1)

    ## Assign all segments to their predicted classification
    for ws in range(num_watersheds):
        clsf_block[watershed_block==ws] = prediction[ws]

    ## Go through each watershed again, and reassign the ones who's size is 
    #   less than 5 pixels. This must be a second loop because the whole
    #   classified raster must be created before we can reassign small segments
    for ws in range(num_watersheds):
        # This is a matrix of True and False, where True corresponds to the 
        #   pixels that have the value of ws
        current_ws = watershed_block==ws
        ws_size = np.sum(current_ws)

        # If an object is smaller than 5 pixels, and completely surrounded by 
        #   a (different) single category, reassign the small object to be the 
        #   same classification as the surrounding area. 
        if ws_size <= 5 and ws_size != 0:
            # Finding the x/y coordinates of the watershed
            index = np.where(current_ws)
            # Reassigns the watershed based on the neighboring pixels
            neighbor_values = neighbor_pixels(clsf_block, index)
            if neighbor_values == 0:
                clsf_block[current_ws] = 0
            elif neighbor_values == 1:
                clsf_block[current_ws] = 1
            elif neighbor_values == 2:
                clsf_block[current_ws] = 2
            elif neighbor_values == 3:
                clsf_block[current_ws] = 3
            elif neighbor_values == 4:
                clsf_block[current_ws] = 4

    # Setting the empty pixels to 0
    clsf_block[image_block==0] = 0

    # Shadow is being reassigned to ice and snow. 
    # clsf_block[clsf_block==5] = 1

    return clsf_block


def neighbor_pixels(image_block, index):
    '''
    Finds the average value of pixels surrounding the given watershed
    '''
    pixel_values = []
    
    top = [index[0][0], index[1][0]]
    bottom = [index[0][-1], index[1][-1]]
    right = [index[0][np.where(index[1] == np.amax(index[1]))], 
             index[1][np.where(index[1] == np.amax(index[1]))]]
    left = [index[0][np.where(index[1] == np.amin(index[1]))], 
            index[1][np.where(index[1] == np.amin(index[1]))]]

    if left[1][0] < 2:
        left[1][0] = 2
    if right[1][0] > 253:
        right[1][0] = 253
    if top[0] < 2:
        top[0] = 2
    if bottom[0] > 253:
        bottom[0] = 253
    pixel_values.append(image_block[left[0][0],left[1][0]-2])
    pixel_values.append(image_block[right[0][0],right[1][0]+2])
    pixel_values.append(image_block[top[0]-2,top[1]])
    pixel_values.append(image_block[bottom[0]+2,bottom[1]])
    
    pixel_average = np.average(pixel_values)
    return pixel_average


def plot_confusion_matrix(y_pred, y):
    plt.imshow(metrics.confusion_matrix(y_pred, y),
                cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.xlabel("true value")
    plt.ylabel("predicted value")
    plt.show()
    print metrics.confusion_matrix(y_pred, y)
    

def main():

    #### Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", 
                        help="directory and filename of image watersheds to be classified")
    parser.add_argument("training_dataset",
                        help="training data file")
    parser.add_argument("training_label", type=str,
                        help="name of training classification list")
    parser.add_argument("-p", "--parallel", metavar='int', type=int, default=1,
                        help='''number of processing threads to create.''')
    parser.add_argument("-q", "--quality", action="store_true",
                        help="print the quality assessment of the training dataset.")
    parser.add_argument("--debug", action="store_true",
                        help="display one classified subimage at a time, with the option of quitting.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display progress text.")

    #### Parse Arguments
    args = parser.parse_args()
    input_filename = args.input_filename
    tds_file = args.training_dataset
    tds_list = args.training_label
    threads = args.parallel
    quality_control = args.quality
    debug_flag = args.debug
    verbose_flag = args.verbose

    ## Load the training data
    tds = utils.load_tds(tds_file,tds_list)

    
    #### Classify the image with inputs
    clsf_im = classify_image(input_filename, tds, threads=threads,
                                                  quality_control=quality_control, 
                                                  debug_flag=debug_flag, 
                                                  verbose=verbose_flag)

    # utils.save_results("classification_results", os.path.dirname(input_filename), output_filename, pixel_counts)

if __name__ == "__main__":
    main()
