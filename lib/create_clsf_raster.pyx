import numpy as np
from ctypes import *
import skimage.morphology as morph


def create_clsf_raster(int [:] prediction,
                       int [:,:,:] intensity_image_view,
                       int [:,:] label_image_view):
    '''
    Transfer classified results from a list of segment:classification pairs
        to a raster where pixel values are the classification result. 
    '''
    cdef int num_ws
    cdef int y,x
    cdef int x_dim, y_dim

    # Create a blank image that we will assign values based on the prediction for each
    #   watershed. 
    clsf_block = np.empty(np.shape(intensity_image_view[:,:,0]), dtype=c_int)
    cdef int [:, :] clsf_block_view = clsf_block
    
    # Watershed indexes start at 0, so we have to add 1 to get the number. 
    num_ws = np.amax(label_image_view) + 1
    x_dim, y_dim = np.shape(intensity_image_view)[0:2]
    # Check to see if the whole block is one segment (why 2?)
    if num_ws >= 2:
        # Assign all segments to their predicted classification
        for y in range(y_dim):
            for x in range(x_dim):
                # Setting the empty pixels (at least 3 bands have values of 0) to 0
                if ((intensity_image_view[x,y,0] == 0)
                    & (intensity_image_view[x,y,1] == 0)
                    & (intensity_image_view[x,y,2] == 0)):
                    clsf_block_view[x,y] = 0
                else:
                    clsf_block_view[x,y] = prediction[label_image_view[x,y]]
    else:
        # Assign the prediction for that one segment to the whole image
        for y in range(y_dim):
            for x in range(x_dim):
                # Set the empty pixels (at least 3 bands have values of 0) to 0
                if ((intensity_image_view[x,y,0] == 0)
                    & (intensity_image_view[x,y,1] == 0)
                    & (intensity_image_view[x,y,2] == 0)):
                    clsf_block_view[x,y] = 0
                else:
                    clsf_block_view[x,y] = prediction[0]

    clsf_block = np.copy(clsf_block_view)

    return clsf_block


def filter_small_segments(clsf_block):
    '''
    Remove small segments from the classified image.
    All regions smaller than the defined structuring element will be removed
      so long as the surrounding classification is a single category.
    '''
    # Structuring element.
    strel = morph.disk(2)

    # Sequentially perform both an opening and closing operation to
    #  remove both 'dark' and 'light' speckle.
    clsf_block_o = morph.opening(clsf_block,strel)
    clsf_block_oc = morph.closing(clsf_block_o,strel)

    return clsf_block_oc