# OSSP
## Open Source Sea-ice Processing
### Open Source Algorithm for Detecting Sea Ice Surface Features in High Resolution Optical Imagery

### Nicholas Wright and Christopher Polashenski

## Introduction

Welcome to OSSP; a set of tools for detecting surface features in high resolution optical imagery of sea ice. The primary focus is on the differentiation of open water, melt ponds, and snow/ice. 

The Anaconda distribution of Python is recommended, but any distribution with the appropriate packages will work. You can download Anaconda, version 2.7, here: https://www.continuum.io/downloads


## Dependencies

* Python 2.7
* gdal (v2.0 or above)
* numpy
* scipy
* h5py
* skimage
* sklearn
* matplotlib
* Tkinter

## Usage

For detailed usage instructions, see the instructional document here: <>

### batch\_process_mp.py

This combines all steps of the image classification scheme into one script. This script finds all appropriately formatted files in the input directory (.tif and .jpg) and queues them for processing. For each image, this script processes them as follows: Image Subdivision (Splitter.py) -> Segmentation (Watershed.py) -> Classification (RandomForest.py) -> Calculate statistics -> Recompile classified splits. batch\_process_mp.py is able to utilize more than one core of the processor for the segmentation and classification phases. 

#### Required Arguments
* __input directory__: directory containing all of the images you wish to process Note that all .jpg and .tif images in the input directory as well as all sub-directories of it will be processed.
* __image type__: {‘srgb’, ‘wv02_ms’, ‘pan'}: the type of imagery you are processing. 
  a. 'srgb': RGB imagery taken by a typical camera
  b. 'wv02_ms': DigitalGlobe WorldView 2 multispectral imagery,
  c. 'pan': High resolution panchromatic imagery
* __training dataset file__: complete filepath of the training dataset you wish to use to analyze the input imagery

#### Optional Arguments

* __-s | --splits__: The number of times to split the input image for improved processing speed. This is rounded to the nearest square number. *Default = 9*.
* __-p | --parallel__: The number of parallel processes to run (i.e. number of cpu cores to utilize). *Default = 1*. 
* __--training\_label__: The label of a custom training dataset. See advanced section for details. *Default = image\_type*.

#### Notes:

Example: batch\_process\_mp.py input\_dir im\_type training\_dataset\_file -s 4 -p 2

This example will process all .tif and .jpg files in the input directory, using the training data found in training\_dataset\_file using two processors, and splitting the image into four sections

In general, images should be divided into parts small enough to easily load into RAM. This depends strongly on the computer running these scripts. Segments should typically not exceed 1 or 2gb in size for best results. For the 5-7mb RGB images provided as test subjects, subdivision is not required (use –s 1 as the optional argument). Processing speed can be increased by combining subdivision with multiple cores. For a full multispectral WorldView scene, which may be 16gb or larger, 9 or 16 segments are typically needed. The number of parallel processes to run should be selected such that num_cores * subdivision filesize << available system ram. 


### Splitter.py

This script reads in a raw image, stretches the pixel intensity values to the full 8-bit range, and subdivides the image into _s_ number of subimages. The output file is in hdf5 format, and is ready to be ready by Watershed.py. 

#### Positional Arguments
* __input_dir__: Directory path of the input image
* __filename__: Name of the image to be split
* __image type__: {‘srgb’, ‘wv02_ms’, ‘pan'}: the type of imagery you are processing. 
a. 'srgb': RGB imagery taken by a typical camera
b. 'wv02_ms': DigitalGlobe WorldView 2 multispectral imagery,
c. 'pan': High resolution panchromatic imagery

#### Optional Arguments
* __--output_dir__: Directory path for output images.
* __-s | --splits__: The number of times to split the input image for improved processing speed. This is rounded to the nearest square number. *Default = 9*.
* __-v | --verbose__: Display text information and progress of the script.


### Watershed.py

This script loads the output of Splitter.py, and segments the image using an edge detection followed by watershed segmentation.

#### Positional Arguments
* __input_dir__: Directory path of the input image.
* __filename__: Name of the segmented image file (Splitter output: .h5).

#### Optional Arguments
* __--output_dir__: Directory path for output files.
* __--histogram__: Display histogram of pixel intensity values before segmentation.
* __-c | --color__: Save a color (rgb) version of the input image.
* __-t | --test__: Inspect segmentation results results before saving output files. 
* __-v | --verbose__: Display text information and progress of the script.


### RandomForest.py

Classified the segmented image (output of Watershed.py) using a Random Forest machine learning algorithm. Training data can be created on a segmented image using the GUI in training_gui.py. 

#### Positional Arguments
* __input\_filename__: Directory and filename of image watersheds to be classified.
* __training\_dataset__: Directory and filename of the training dataset (.h5)
* __training\_label__: name of training classification list

#### Optional Arguments
* __-q | --quality__: Display a quality assessment (OOB score and attribute importance) of the training dataset.
* __--debug__: Display one classified subimage at a time, with the option of quitting after each.
* __-v | --verbose__: Display text information and progress of the script.


### training_gui.py

#### Positional Arguments:
* __input__: In mode 1 this is a folder containing the training images. In mode 2, the input is a classified image file (.h5).
* __image type__: {‘srgb’, ‘wv02_ms’, ‘pan'}: the type of imagery you are processing. 
a. 'srgb': RGB imagery taken by a typical camera
b. 'wv02_ms': DigitalGlobe WorldView 2 multispectral imagery,
c. 'pan': High resolution panchromatic imagery
* __-m | --mode__: {1,2}. How you would like to the training GUI. 1: create a training dataset from folder of raw images. 2: assess the accuracy of a classified image (output of RandomForest.py).

#### Optional arguments:
* __--tds_file__: Only used for mode 1. Existing training dataset file. Will create a new one with this name if none exists. *Default = <\image_type>\_training\_data.h5*.
* __-s | --splits__: The number of times to split the input image for improved processing speed. This is rounded to the nearest square number. *Default = 9*.

### Contact
Nicholas Wright

