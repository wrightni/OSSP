# title: Random Forest Classifier
# author: Nick Wright

import argparse
from multiprocessing import Process, Queue
from ctypes import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

from lib import utils, debug_tools
from lib import attribute_calculations as attr_calc
from lib import create_clsf_raster as ccr

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

    # Method for assessing the quality of the training dataset.
    # quality_control = True
    # if quality_control == True:
    #     debug_tools.test_training(label_vector, training_feature_matrix)
    #     aa = raw_input("Continue? ")
    #     if aa == 'n':
    #         quit()

    #### Construct the random forest decision tree using the training data set
    rfc = RandomForestClassifier(n_estimators=100)
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

    # Cast data as C int.
    image_block = np.ndarray.astype(image_block, c_int)
    watershed_block = np.ndarray.astype(watershed_block, c_int)

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
        input_feature_matrix = attr_calc.analyze_ms_image(
                                image_block, watershed_block)
    elif image_type == 'srgb':
        input_feature_matrix = attr_calc.analyze_srgb_image(image_block,watershed_block)
    elif image_type == 'pan':
        input_feature_matrix = attr_calc.analyze_pan_image(
                                image_block, watershed_block, image_date)

    input_feature_matrix = np.array(input_feature_matrix)

    # Predict the classification of each segment
    ws_predictions = rfc.predict(input_feature_matrix)
    ws_predictions = np.ndarray.astype(ws_predictions,dtype=c_int)
    # Create the classified image by replacing watershed id's with 
    #   classification values.
    # If there is more than one band, we have to select one (using 2 for 
    #   no particular reason).
    # if image_type == 'pan':
    #     clsf_block = create_clsf_raster(ws_predictions, watershed_block, 
    #                                     image_block)

    clsf_block = ccr.create_clsf_raster(ws_predictions, image_block,
                                        watershed_block)
    # clsf_block = ccr.filter_small_segments(clsf_block)
    return clsf_block


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

    # input_image, watershed_data = read_inputfile()
    
    #### Classify the image with inputs
    # clsf_im = (input_image, watershed_data, training_dataset, meta_data,
    #                    threads=1, quality_control=False, verbose=False):
    # clsf_im = classify_image(input_filename, tds, threads=threads,
    #                                               quality_control=quality_control,
    #                                               debug_flag=debug,
    #                                               verbose=verbose_flag)

    # utils.save_results("classification_results", os.path.dirname(input_filename), output_filename, pixel_counts)

if __name__ == "__main__":
    main()
