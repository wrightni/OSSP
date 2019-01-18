# title: Random Forest Classifier
# author: Nick Wright

import argparse
from multiprocessing import Process, Queue
from ctypes import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import time

from lib import utils, debug_tools
from lib import attribute_calculations as attr_calc
from lib import create_clsf_raster as ccr

# tqdm for progress bar


def classify_image(input_image, watershed_data, training_dataset, meta_data):
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
    # num_blocks = len(input_image[1])
    num_bands = np.shape(input_image)[0]
    image_type = meta_data[0]
    image_date = meta_data[1]

    ## Restructure the input data.
    # We are creating a single list where each element of the list is one
    #   block (old: subimage) of the image and is a stack of all bands.  
    # image_data = []    # [block:row:column:band]
    # for blk in range(num_blocks):
    #     image_data.append(utils.create_composite(
    #             [input_image[b][blk] for b in range(1,num_bands+1)]))
    # input_image = None
    image_data = utils.create_composite([input_image[b] for b in range(num_bands)])

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

    clsf_block = classify_block(image_data, watershed_data, image_type, image_date, rfc)

    return clsf_block


def classify_block(image_block, watershed_block, image_type, image_date, rfc):

    # Cast data as C int.
    image_block = np.ndarray.astype(image_block, c_int)
    watershed_block = np.ndarray.astype(watershed_block, c_int)
    # print(np.shape(image_block))

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
