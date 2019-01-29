# OSSP
## Open Source Sea-ice Processing
### Open Source Algorithm for Detecting Sea Ice Surface Features in High Resolution Optical Imagery

### Nicholas Wright and Christopher Polashenski

## Introduction

Welcome to OSSP; a set of tools for detecting surface features in high resolution optical imagery of sea ice. The primary focus is on the detection of and differentiation between open water, melt ponds, and snow/ice. 

The Anaconda distribution of Python is recommended, but any distribution with the appropriate packages will work. You can download Anaconda, version 2.7, here: https://www.continuum.io/downloads


## Dependencies

* gdal (v2.0 or above)
* numpy
* scipy
* h5py
* scikit-image
* sklearn
* matplotlib
* Tkinter

#### Optional
* tqdm (for progress bar)
* PGC imagery_utils (for WV pansharpening)

## Usage

For detailed usage and installation instructions, see the pdf document 'Algorithm_Instructions.pdf'

### setup.py

The first step is to run the setup.py script to compile C libraries. Run __python setup.py build\_ext --build-lib .__ from the OSSP directory. Be sure to include the period after --build-lib. 

### ossp_process.py

This combines all steps of the image classification scheme into one script and should be the primary script to use. If given a folder of images, this script finds all appropriately formatted files directory (.tif(f) and .jpg) and queues them for processing. If given an image file, this script processes that single image alone. This script processes images as follows: Image preprocessing (histogram stretch or pansharpening if chosen) -> segmentation (segment.py) -> classification (classify.py) -> calculate statistics.

#### Required Arguments
* __input directory__: directory containing all of the images you wish to process Note that all .jpg and .tif images in the input directory as well as all sub-directories of it will be processed. Can also provide the path and filename to a single image to process only that image.
* __image type__: {‘srgb’, ‘wv02_ms’, ‘pan'}: the type of imagery you are processing. 
  1. 'srgb': RGB imagery taken by a typical camera
  2. 'wv02_ms': DigitalGlobe WorldView 2 multispectral imagery
  3. 'pan': High resolution panchromatic imagery
* __training dataset file__: complete filepath of the training dataset you wish to use to analyze the input imagery

#### Optional Arguments

* __-o | --output_dir__: Directory to write output files. 
* __-v | --verbose__: Display text output as algorithm progresses. 
* __-c | --stretch__: {'hist', 'pansh', 'none'}: Apply an image correction prior to classification. Pansharpening / orthorectification option requires PGC scripts. *Default = hist*.
* __--pgc_script__: Path for the PGC imagery_utils folder if 'pansh' was chosen for the image correction.
* __--training\_label__: The label of a custom training dataset. See advanced section for details. *Default = image\_type*.

#### Notes:

Example: ossp\_process.py input\_dir im\_type training\_dataset\_file -v

This example will process all .tif and .jpg files in the input directory.


### segment.py

This script segments the image using an edge detection followed by watershed segmentation.

#### Positional Arguments
* __input_dir__: Directory path of the input image.
* __filename__: Name of the segmented image file (Splitter output: .h5).

#### Optional Arguments
* __--output_dir__: Directory path for output files.
* __--histogram__: Display histogram of pixel intensity values before segmentation.
* __-c | --color__: Save a color (rgb) version of the input image.
* __-t | --test__: Inspect segmentation results results before saving output files. 
* __-v | --verbose__: Display text information and progress of the script.


### classify.py

Classified the segmented image (output of segment.py) using a Random Forest machine learning algorithm. Training data can be created on a segmented image using the GUI in training_gui.py. 

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
  1. 'srgb': RGB imagery taken by a typical camera
  2. 'wv02_ms': DigitalGlobe WorldView 2 multispectral imagery,
  3. 'pan': High resolution panchromatic imagery
* __-m | --mode__: {1,2}. How you would like to the training GUI. 1: create a training dataset from folder of raw images. 2: assess the accuracy of a classified image (output of RandomForest.py).

#### Optional arguments:
* __--tds_file__: Only used for mode 1. Existing training dataset file. Will create a new one with this name if none exists. *Default = <image_type>\_training\_data.h5*.
* __-s | --splits__: The number of times to split the input image for improved processing speed. This is rounded to the nearest square number. *Default = 9*.


### Contact
Nicholas Wright

