#title: Random Forest Classifier
#author: Nick Wright

#purpose: Loads a pansharpened satellite image, a watershed transform of the same image,
#			and a training data set (created by ONR GUI.py) and classifies the watershed
#			superpixels as ice, melt ponds, or open water. 

import argparse
import os
import h5py
import csv
import time
import math
import sys	
from random import randint, uniform, gauss
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mimg
from skimage import filters, morphology, feature, exposure, measure, segmentation
from scipy.misc import bytescale
from skimage.measure import perimeter
from skimage.filters.rank import entropy
from skimage.morphology import disk

from lib import utils, feature_calculations

def main():

	#### Set Up Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("input_filename", 
						help="directory and filename of image watersheds to be classified")
	parser.add_argument("training_dataset",
						help="training data file")
	parser.add_argument("training_label", type=str,
						help="name of training classification list")
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
	quality_control = args.quality
	debug_flag = args.debug
	verbose_flag = args.verbose

	## Load the training data
	tds = utils.load_tds(tds_file,tds_list)

	#### Classify the image with inputs
	output_filename, pixel_counts = process(input_filename, tds, quality_control=quality_control, debug_flag=debug_flag, verbose=verbose_flag)

	utils.save_results("classification_results", os.path.dirname(input_filename), output_filename, pixel_counts)

def process(input_filename, training_dataset, output_filepath=False, quality_control=False, debug_flag=False, verbose=False):
	'''
	Pulls training data from the directory containing watershed segments if it exists, otherwise
	searches for training data in the root directory, containing all of the date directories. 
	'''

	# Parse the training_dataset input
	label_vector = training_dataset[0]
	training_feature_matrix = training_dataset[1]

	# Set the output filepath to the input one if there is no path argument provided 
	if output_filepath == False:
		output_filepath = os.path.dirname(input_filename)
	# Removes the '_watersheds.h5' portion of the input filename
	output_filename = os.path.split(input_filename)[1][:-13]
	
	# Loading the image data
	input_file = h5py.File(input_filename, 'r')

	input_image = [] 	#[subimage:row:column:band]
	band_1 = input_file['original_1'][:]
	watershed_image = input_file['watershed'][:]
	watershed_dimensions = input_file['dimensions'][:]
	num_x_subimages = watershed_dimensions[0]
	num_y_subimages = watershed_dimensions[1]
	
	im_type = input_file.attrs.get('Image Type')
	im_date = input_file.attrs.get('Image Date')

	# Use this to read files created before the image type attribute change
	# im_type = 'wv02_ms'

	#Method for assessing the quality of the training dataset. 
	if quality_control == True:
		test_training(label_vector, training_feature_matrix)
		aa = raw_input("Continue? ")
		if aa == 'n':
			quit()

	# If there is no information in this image file, save a dummy classified image and exit
	# This can often happen depending on the original image dimensions and the amount it was split
	if np.sum(band_1) == 0:
		classified_image_path = os.path.join(output_filepath, output_filename + '_classified_image.png')
		outfile = h5py.File(os.path.join(output_filepath, output_filename + '_classified.h5'),'w')
		
		if im_type == 'wv02_ms':
        		empty_bands = np.zeros(np.shape(band_1)[0],np.shape(band_1)[1],8)
                        empty_image = utils.compile_subimages(empty_bands, num_x_subimages, num_y_subimages, 8)
                elif im_type == 'srgb':
                        empty_bands = np.zeros(np.shape(band_1)[0],np.shape(band_1)[1],3)
                        empty_image = utils.compile_subimages(empty_bands, num_x_subimages, num_y_subimages, 3)
                elif im_type == 'pan':
                        empty_image = np.zeros(np.shape(band_1))
		
		outfile.create_dataset('classified', data=empty_image,compression='gzip',compression_opts=9)
		outfile.create_dataset('original', data=empty_image,compression='gzip',compression_opts=9)
		outfile.close()
                # return a 1x5 array with values of one for the pixel counts
		return output_filename, np.ones(5)

	# Adds remaining 7 bands if they exist
	if im_type == 'wv02_ms':

		if verbose: print "Reconstructing multispectral image... "
		band_2 = input_file['original_2'][:]
		band_3 = input_file['original_3'][:]
		band_4 = input_file['original_4'][:]
		band_5 = input_file['original_5'][:]
		band_6 = input_file['original_6'][:]
		band_7 = input_file['original_7'][:]
		band_8 = input_file['original_8'][:]

		for i in range(len(band_1)):
			input_image.append(create_composite([
				band_1[i], band_2[i], band_3[i], band_4[i], 
				band_5[i], band_6[i], band_7[i], band_8[i]
		]))
		if verbose: print "Done. "

	elif im_type == 'srgb':
		if verbose: print "Reconstructing multispectral image... "
		band_2 = input_file['original_2'][:]
		band_3 = input_file['original_3'][:]

		for i in range(len(band_1)):
			input_image.append(create_composite([
				band_1[i], band_2[i], band_3[i]
		]))
		if verbose: print "Done. "

	elif im_type == 'pan':
		input_image = band_1

	input_file.close()

	#Constructing the random forest tree based on the training data set and labels
	rfc = RandomForestClassifier()
	rfc.fit(training_feature_matrix, label_vector)

	#Using the random forest tree to classify the input image.
	#Runs an analysis on each subimage in the watershed_image list

	if verbose:
		start_time = time.clock() 
		prog1 = 0
		prog2 = 10
		print "Predicting Image..."

	classified_image = []
	for subimage in range(len(watershed_image)):

		if verbose:
			## Progress tracker
			if int(float(prog1)/float(len(watershed_image))*100) == prog2:
				print "%s Percent" %prog2
				prog2 += 10
			prog1 += 1
			subimage_start_time = time.clock()

		if debug_flag == True:		
			if subimage < 100:
				classified_image.append(np.zeros(np.shape(input_image[subimage])[0:2]))
				continue

		cur_image = np.copy(input_image[subimage])
		cur_ws = np.copy(watershed_image[subimage])

		## Skip any subimages that contain no data, and set the classification values to 0
		if np.amax(cur_image) < 2:
			classified_image.append(np.zeros(np.shape(cur_image)[0:2]))
			continue

		# If the entire image is a single watershed, we have to handle the neighboring region
		#	calculation specially. This assignes the neighboring regions values to be the same
		#	as the internal values. 
		if np.amax(cur_ws) == 1 and im_type == 'pan':
			entropy_image = entropy(bytescale(cur_image), disk(4))
			features = []

			#Average Pixel Value
			features.append(np.average(cur_image))
			#Pixel Median
			features.append(np.median(cur_image))
			#Pixel min
			features.append(np.amin(cur_image))
			#Pixel max
			features.append(np.amax(cur_image))
			#Standard Deviation
			features.append(np.std(cur_image))
			#Size of Superpixel 
			features.append(len(cur_image))

			features.append(np.average(entropy_image))

			#"Neighbor" average
			features.append(np.average(cur_image))
			#"Neighbor" std
			features.append(np.std(cur_image))
			#"Neighbor" max
			features.append(np.max(cur_image))
			#"Neighbor" entropy
			features.append(np.average(entropy_image))

			# Date
			features.append(int(im_date))

			input_features = features
			input_features = np.array(input_features).reshape(1,-1)
			ws_pred = rfc.predict(input_features)
			classified_image.append(plotPrediction(ws_pred, cur_ws, cur_image))
			continue


		# We need the superpixel labels to start at 0. This shifts the entire label image down so that
		# the first label is 0, if it isn't already. 
		if np.amin(cur_ws) > 0:
			cur_ws -= np.amin(cur_ws)

		if im_type == 'wv02_ms':
			input_feature_matrix = feature_calculations.analyze_ms_image(cur_image, cur_ws)
		elif im_type == 'srgb':
			entropy_image = entropy(bytescale(cur_image[:,:,0]), disk(4))
			input_feature_matrix = feature_calculations.analyze_srgb_image(cur_image, cur_ws, entropy_image)
		elif im_type == 'pan':
			entropy_image = entropy(bytescale(cur_image), disk(4))
			input_feature_matrix = feature_calculations.analyze_pan_image(cur_image, cur_ws, entropy_image, im_date)

		input_feature_matrix = np.array(input_feature_matrix)

		# Predict the classification based on each input feature list
		ws_pred = rfc.predict(input_feature_matrix)

		# Create the classified image by replacing watershed id's with classification values.
		# If there is more than one band, we have to select one (using 2 for no particular reason).
		if im_type == 'pan':
			classified_image.append(plotPrediction(ws_pred, cur_ws, cur_image))
		else:
			classified_image.append(plotPrediction(ws_pred, cur_ws, cur_image[:,:,2]))

		if verbose:	
			subimage_time = time.clock() - subimage_start_time
			print str(subimage+1) + "/" + str(len(watershed_image)) + " Time: " + str(subimage_time)
		
		## Display one classification result at a time if the --debug flag was input.
		if debug_flag == True:

			if im_type == 'pan':
				display_image(cur_image,cur_ws,classified_image[subimage],1)
			else:
				display_image(cur_image[:,:,1],cur_ws,classified_image[subimage],1)

			sum_snow, sum_gray_ice, sum_melt_ponds, sum_open_water, sum_shadow = utils.count_features(classified_image[subimage])
			print "Number Snow: %i" %(sum_snow)
			print "Number Pond: %i" %(sum_melt_ponds)
			print "Number Gray Ice: %i" %(sum_gray_ice)
			print "Number Water: %i" %(sum_open_water)
			print "Number Shadow: %i" %(sum_shadow)

			aa = raw_input("Another? ")
			if aa == 'n':
				quit()

	if verbose:	
		elapsed_time = time.clock() - start_time
		print "Done. "
		print "Time elapsed: {0}".format(elapsed_time)

	compiled_classified = utils.compile_subimages(classified_image, num_x_subimages, num_y_subimages, 1)
	if im_type == 'wv02_ms':
		compiled_original = utils.compile_subimages(input_image, num_x_subimages, num_y_subimages, 8)
	elif im_type == 'srgb':
		compiled_original = utils.compile_subimages(input_image, num_x_subimages, num_y_subimages, 3)
	elif im_type == 'pan': 
		compiled_original = utils.compile_subimages(input_image, num_x_subimages, num_y_subimages, 1)

	if verbose: print "Saving..."

	classified_image_path = os.path.join(output_filepath, output_filename + '_classified_image.png')
	utils.save_color(compiled_classified, classified_image_path)
	
	with h5py.File(os.path.join(output_filepath, output_filename + '_classified.h5'),'w') as outfile:
		outfile.create_dataset('classified', data=compiled_classified,compression='gzip',compression_opts=9)
		outfile.create_dataset('original', data=compiled_original ,compression='gzip',compression_opts=9)

	#### Count the number of pixels that were in each classification category. 
	sum_snow, sum_gray_ice, sum_melt_ponds, sum_open_water, sum_shadow = utils.count_features(compiled_classified)
	pixel_counts = [sum_snow, sum_gray_ice, sum_melt_ponds, sum_open_water, sum_shadow]

	# Clear the image datasets from memory
	compiled_classified = None
	input_image = None
	watershed_image = None

	cur_image = None
	cur_ws = None
	entropy_image = None
	
	if verbose: print "Done."

	return output_filename, pixel_counts

def plot_confusion_matrix(y_pred, y):
	plt.imshow(metrics.confusion_matrix(y_pred, y),
				cmap=plt.cm.binary, interpolation='nearest')
	plt.colorbar()
	plt.xlabel("true value")
	plt.ylabel("predicted value")
	plt.show()
	print metrics.confusion_matrix(y_pred, y)
	
	
# Combines multiple bands (RBG) into one 3D array
# Adapted from:  http://gis.stackexchange.com/questions/120951/merging-multiple-16-bit-image-bands-to-create-a-true-color-tiff
# Useful band combinations: http://c-agg.org/cm_vault/files/docs/WorldView_band_combs__2_.pdf
def create_composite(band_list):

	img_dim = band_list[0].shape
	img = np.zeros((img_dim[0], img_dim[1], len(band_list)), dtype=np.uint8)
	for i in range(len(band_list)):
		img[:,:,i] = band_list[i]
	
	return img

def plotPrediction(prediction, watershed_subimage, raw_subimage):

	#Create a blank image that we will assign values based on the prediction for each
	#	watershed. 
	pred_image = np.zeros(np.shape(watershed_subimage),dtype=np.uint8)
	#Check to see if the whole subimage is one superpixel
	if np.amax(watershed_subimage) == 1:
		pred_image = pred_image + prediction[0]
		pred_image[raw_subimage == 0] = 0
		return pred_image

	num_watersheds = int(np.amax(watershed_subimage)+1)

	# Assign all watersheds to the predicted value returned by the random
	# forest method
	for ws in range(num_watersheds):
		pred_image[watershed_subimage==ws] = prediction[ws]

	# Go through each watershed again, and reassign the ones who's size is 
	# less than 5 pixels.
	for ws in range(num_watersheds):
		#This is a matrix of True and False, where True corresponds to the 
		# pixels that have the value of ws
		current_ws = watershed_subimage==ws

		#Check to see if the watershed is less than 5 but greater than 0 in size, and the raw 
		# subimage is not zero,	meaning we want to reclassify the ws to the surrounding class
		#  is this needed?: and np.sum(raw_subimage[current_ws]) != 0
		if np.sum(current_ws) < 5 and np.sum(current_ws) != 0:
			# if np.sum(current_ws)==0:
			# 	continue
			#Finding the x/y coordinates of the watershed
			index = np.where(current_ws)
			#Reassigns the watershed based on the neighboring pixels
			neighbor_values = neighbor_pixels(pred_image, index)
			if neighbor_values == 0:
				pred_image[current_ws] = 0
			elif neighbor_values == 1:
				pred_image[current_ws] = 1
			elif neighbor_values == 2:
				pred_image[current_ws] = 2
			elif neighbor_values == 3:
				pred_image[current_ws] = 3
			elif neighbor_values == 4:
				pred_image[current_ws] = 4

				
	# Setting the empty pixels to 0
	pred_image[raw_subimage==0] = 0

	# Shadow is being reassigned to ice and snow. 
	pred_image[pred_image==5] = 1

	return pred_image

def neighbor_pixels(subimage, index):
	
	pixel_values = []
	
	top = [index[0][0], index[1][0]]
	bottom = [index[0][-1], index[1][-1]]
	right = [index[0][np.where(index[1] == np.amax(index[1]))], index[1][np.where(index[1] == np.amax(index[1]))]]
	left = [index[0][np.where(index[1] == np.amin(index[1]))], index[1][np.where(index[1] == np.amin(index[1]))]]

	if left[1][0] < 2:
		left[1][0] = 2
	if right[1][0] > 253:
		right[1][0] = 253
	if top[0] < 2:
		top[0] = 2
	if bottom[0] > 253:
		bottom[0] = 253
	pixel_values.append(subimage[left[0][0],left[1][0]-2])
	pixel_values.append(subimage[right[0][0],right[1][0]+2])
	pixel_values.append(subimage[top[0]-2,top[1]])
	pixel_values.append(subimage[bottom[0]+2,bottom[1]])
	
	pixel_average = np.average(pixel_values)
	return pixel_average

def display_image(raw,watershed,classified,type):

	# Save a color 
	empty_color = [.1,.1,.1]		#Almost black
	snow_color = [.9,.9,.9]			#Almost white
	pond_color = [.31,.431,.647]	#Blue
	gray_color = [.65,.65,.65]			#Gray
	water_color = [0.,0.,0.]		#Black
	shadow_color = [.100, .545, .0]#Orange

	custom_colormap = [empty_color,snow_color,gray_color,pond_color,water_color,shadow_color]
	custom_colormap = colors.ListedColormap(custom_colormap)

	#Making sure there is atleast one of every pixel so the colors map properly (only changes
	# display image, not saved data)
	classified[0][0] = 0
	classified[1][0] = 1
	classified[2][0] = 2
	classified[3][0] = 3
	classified[4][0] = 4
	classified[5][0] = 5

	# Creating the watershed display image with borders highlighted
	ws_bound = segmentation.find_boundaries(watershed)
	ws_display = create_composite([raw,raw,raw])
	ws_display[:,:,0][ws_bound] = 255
	ws_display[:,:,1][ws_bound] = 255
	ws_display[:,:,2][ws_bound] = 22
	
	# Figure that show 3 images: raw, segmented, and classified
	if type == 1:
		fig, axes = plt.subplots(1,3,subplot_kw={'xticks':[], 'yticks':[]})
		fig.subplots_adjust(left=0.05,right=0.99,bottom=0.05,top=0.90,wspace=0.02,hspace=0.2)

		tnrfont = {'fontname':'Times New Roman'}

		axes[0].imshow(raw,cmap='gray',interpolation='None')
		axes[0].set_title("Raw Image", **tnrfont)
		axes[1].imshow(ws_display,interpolation='None')
		axes[1].set_title("Image Segments", **tnrfont)
		axes[2].imshow(classified,cmap=custom_colormap,interpolation='None')
		axes[2].set_title("Classification Output", **tnrfont)

	# Figure that shows 2 images: raw and classified. 
	if type == 2:
		fig, axes = plt.subplots(1,2,subplot_kw={'xticks':[], 'yticks':[]})
		fig.subplots_adjust(hspace=0.3,wspace=0.05)

		axes[0].imshow(raw,cmap='gray',interpolation='None')
		axes[0].set_title("Raw Image")
		axes[1].imshow(classified,cmap=custom_colormap,interpolation='None')
		axes[1].set_title("Classification Output")

	plt.show()

# Method to assess the training set and classification tree used for this classification
def test_training(label_vector, training_feature_matrix):

		print "Size of training set: %i" %len(label_vector)

		rfc = RandomForestClassifier(oob_score=True)
		rfc.fit(training_feature_matrix, label_vector)
		print "OOB Score: %f" %rfc.oob_score_

		training_feature_matrix = np.array(training_feature_matrix)
		importances = rfc.feature_importances_
		std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
					axis=0)

		feature_names = ['Mean Intensity','Median\nIntensity','Standard\nDeviation','Segment Size','Entropy',
							'Neighbor\nMean Intensity','Neighbor\nStandard\nDeviation','Neighbor\nMaximum',
							'Neighbor\nEntropy','Date']

		feature_names = range(len(training_feature_matrix))

		indices = np.argsort(importances)[::-1]

		# feature_names = ['Mean Intensity','Standard Deviation','Size','Entropy','Neighbor Mean Intensity'
		# 					'Neighbor Standard Deviation','Neighbor Maximum Intensity','Neighbor Entropy','Date']

		# Print the feature ranking
		print("Feature ranking:")

		feature_names_sorted = []
		for f in range(training_feature_matrix.shape[1]):
		 	print("%d. feature %s (%f)" % (f+1, feature_names[indices[f]], importances[indices[f]]))
		 	feature_names_sorted.append(feature_names[indices[f]])

		# Plot the feature importances of the forest
		plt.figure()
		plt.title("Feature importances")
		plt.bar(range(training_feature_matrix.shape[1]), importances[indices],
				color=[.161,.333,.608], yerr=std[indices], align="center", 
				error_kw=dict(ecolor=[.922,.643,.173], lw=2, capsize=3, capthick=2))
		plt.xticks(range(training_feature_matrix.shape[1]), feature_names_sorted)#, rotation='45')
		plt.xlim([-1, training_feature_matrix.shape[1]])
		plt.show()
	
if __name__ == "__main__":
	main()
