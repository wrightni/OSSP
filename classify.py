# title: Random Forest Classifier
# author: Nick Wright

from ctypes import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

from lib import utils
from lib import attribute_calculations as attr_calc
from lib import create_clsf_raster as ccr


def classify_image(input_image, watershed_data, training_dataset, meta_data, wb_ref, bp_ref):
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
    image_type = meta_data[0]
    image_date = meta_data[1]

    ## Parse training_dataset input
    label_vector = training_dataset[0]
    training_feature_matrix = training_dataset[1]

    #### Construct the random forest decision tree using the training data set
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(training_feature_matrix, label_vector)

    clsf_block = classify_block(input_image, watershed_data, image_type, image_date, rfc, wb_ref, bp_ref)

    return clsf_block


def classify_block(image_block, watershed_block, image_type, image_date, rfc, wb_ref, bp_ref):

    # Cast data as C int.
    watershed_block = watershed_block.astype(c_uint32, copy=False)

    ## If the block contains no data, set the classification values to 0
    if np.amax(image_block) < 2:
        clsf_block = np.zeros(np.shape(image_block)[1:3])
        return clsf_block

    ## We need the object labels to start at 0. This shifts the entire 
    #   label image down so that the first label is 0, if it isn't already. 
    if np.amin(watershed_block) > 0:
        watershed_block -= np.amin(watershed_block)
    ## Calculate the features of each segment within the block. This 
    #   calculation is unique for each image type. 
    if image_type == 'wv02_ms':
        input_feature_matrix = attr_calc.analyze_ms_image(image_block, watershed_block,
                                                          wb_ref, bp_ref)
    elif image_type == 'srgb':
        input_feature_matrix = attr_calc.analyze_srgb_image(image_block,watershed_block)
    elif image_type == 'pan':
        input_feature_matrix = attr_calc.analyze_pan_image(
                                image_block, watershed_block, image_date)

    input_feature_matrix = np.array(input_feature_matrix)

    # Predict the classification of each segment
    ws_predictions = rfc.predict(input_feature_matrix)
    ws_predictions = np.ndarray.astype(ws_predictions, dtype=c_int, copy=False)
    # Create the classified image by replacing watershed id's with
    #   classification values.
    # If there is more than one band, we have to select one (using 2 for 
    #   no particular reason).

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
    print(metrics.confusion_matrix(y_pred, y))
