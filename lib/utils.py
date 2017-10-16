import h5py
import os
import csv
import math
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mimg


#### Load Training Dataset (TDS) (Label Vector and Feature Matrix)
def load_tds(file_name, list_name):
	'''
	INPUT: 
		input_directory of .h5 training data
		file_name of .h5 training data
		list_name of label vector contained within file_name
	RETURNS:
		tds = [label_vector, training_feature_matrix]
	'''

	## Load the training data
	with h5py.File(file_name, 'r') as training_file:
		label_vector = training_file[list_name][:]
		segments = training_file['segment_list'][:]
		training_feature_matrix = training_file['feature_matrix'][:]

	## Convert inputs to python lists
	label_vector = label_vector.tolist()
	training_feature_matrix = training_feature_matrix.tolist()
	# Remove feature lists that dont have an associated label
	training_feature_matrix = training_feature_matrix[:len(label_vector)]
	# print "__"
	# print len(label_vector)
	## Remove the segments labeled "unknown" (0)
	while 0 in label_vector:
		i = label_vector.index(0)
		label_vector.pop(i)
		training_feature_matrix.pop(i)

	## Remove the segments labeled "mixed" (0)
	while 6 in label_vector:
		i = label_vector.index(6)
		label_vector.pop(i)
		training_feature_matrix.pop(i)

	## Remove the segments labeled "Shadow" (0)
	while 5 in label_vector:
		i = label_vector.index(5)
		label_vector.pop(i)
		training_feature_matrix.pop(i)

	# Combine the label vector and training feature matrix into one variable. 
	tds = [label_vector,training_feature_matrix]

	return tds

#### Save classification results
def save_results(csv_name, path, image_name, pixel_counts):
	'''
	INPUT: 
		path: location where the output will write
		image_name: name of the image that was classified
		pixel_clounts: number of pixels in each classification category

	Saves a csv with the input information. Appends to existing csv if one already exists

	NOTES:
		Only works with 5 classification categories: 
			[white ice, gray ice, melt ponds, open water, shadow]
	'''

	csv_name = os.path.splitext(csv_name)[0] + ".csv"

	num_pixels = 0
	for i in range(len(pixel_counts)):
		num_pixels +=  pixel_counts[i]
	percentages = []
	for i in range(len(pixel_counts)):
		percentages.append(float(pixel_counts[i]/num_pixels))

	try:
		output_csv = os.path.join(path, csv_name)
		if not os.path.isfile(output_csv):
			with open(output_csv, "wb") as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(["Source", "White Ice", "Gray Ice", "Melt Ponds", "Open Water", "Shadow",
					"Prcnt White Ice", "Prcnt Gray Ice", "Prcnt Melt Ponds", "Prcnt Open Water", "Prcnt Shadow"])
				writer.writerow([image_name, pixel_counts[0], pixel_counts[1], pixel_counts[2], pixel_counts[3], pixel_counts[4],
					percentages[0], percentages[1], percentages[2], percentages[3], percentages[4]])

		else:
			with open(output_csv, "ab+") as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow([image_name, pixel_counts[0], pixel_counts[1], pixel_counts[2], pixel_counts[3], pixel_counts[4],
					percentages[0], percentages[1], percentages[2], percentages[3], percentages[4]])
	except:
		print "error saving csv"
		print pixel_counts


#### Recombine classified image splits
def stitch(image_files, save_path=None):
	'''
	INPUT:
		image_files: list of the image splits for recombination
	RETURN:
		full_classification: full image stitched back together

	NOTES:
		Currently only implemented to recombine classified images, but the
		method could work with any image data.
		There are two levels of recombination. Recompiling the subimages (see
		method below) and recompiling the splits (this method)
	'''

	# Check to see if we have a square number of images
	#	This method relies on floating point precision, but
	#	will be accurate within the scope of this method
	root = math.sqrt(len(image_files))
	if int(root) != root:
		print "Incomplete set of images!"
		return None

	original_list = []
	classified_list = []

	image_files.sort()

	## Read the classified data and the original image data
	for image in image_files:
		with h5py.File(image,'r') as inputfile:
			original_image = inputfile['original'][:]
			original_list.append(original_image)
			classified_image = inputfile['classified'][:]
			classified_list.append(classified_image)


	# Find the right dimensions for stitching the images back together
	box_side = int(math.sqrt(len(classified_list)))
	# Get the number of bands based on the input list
	try: 
		num_bands = np.shape(original_list)[3]
	except IndexError:
		num_bands = 1
	# Stitch the original back together
	full_original = compile_subimages(original_list,box_side,box_side,num_bands)
	# Stitch the classified image back together
	full_classification = compile_subimages(classified_list,box_side,box_side)

	if os.path.isdir(save_path):
		output_name = os.path.join(save_path, os.path.split(image_files[0])[1][:-18])
		save_color(full_classification, output_name + "_classified_image.png")
		fout = h5py.File(output_name + "_classified.h5",'w')
		fout.create_dataset('classified',data=full_classification,compression='gzip',compression_opts=9)
		fout.create_dataset('original',data=full_original,compression='gzip',compression_opts=9)
		fout.close()
	else:
		save_color(full_classification, image_files[0][:-18] + "_classified_image.png")
		fout = h5py.File(image_files[0][:-18] + "_classified.h5",'w')
		fout.create_dataset('classified',data=full_classification,compression='gzip',compression_opts=9)
		fout.create_dataset('original',data=full_original,compression='gzip',compression_opts=9)
		fout.close()

	return full_classification


#### Compiles the subimages of a split
def compile_subimages(subimage_list, num_x_subimages, num_y_subimages, bands=1):
	'''
	INPUT:
		subimage_list: the list of subimages, in left to right top to bottom order
		num_x_subimages: number of subimages in the x dimension
		num_y_subimages: number of subimages in the y dimension
		bands: number of spectral bands of the input image
	RETURNS:
		compiled_image: single [x,y,b] image
	'''
	x_size = np.shape(subimage_list[0])[1]
	y_size = np.shape(subimage_list[0])[0]

	if bands != 1:
		compiled_image = np.zeros([num_y_subimages*y_size,num_x_subimages*x_size,bands],dtype='uint8')
		counter = 0
		for y in range(num_y_subimages):
			for x in range(num_x_subimages):
				compiled_image[y*y_size:(y+1)*y_size, x*x_size:(x+1)*x_size, :] = subimage_list[counter]
				counter += 1
	else:
		compiled_image = np.zeros([num_y_subimages*y_size,num_x_subimages*x_size],dtype='uint8')
		counter = 0
		for y in range(num_y_subimages):
			for x in range(num_x_subimages):
				compiled_image[y*y_size:(y+1)*y_size, x*x_size:(x+1)*x_size] = subimage_list[counter]
				counter += 1

	return compiled_image

#### Saves an image with custom colormap
def save_color(image, save_name, custom_colormap=False):
	''''
	INPUTS:
		image: The image you want to save
		save_name: full name and filepath where you want the image to go
		custom_colormap: matplotlib colormap if you want to use your own
			defaults to nwright's 5 class colormap
	
	Saves a .png of the input image with desired colormap
	'''

	if custom_colormap is False:
		# Colors for the output image
		empty_color = [.1,.1,.1]		#Almost black
		snow_color = [.9,.9,.9]			#Almost white
		pond_color = [.31,.431,.647]	#Blue
		gray_color = [.5,.5,.5]			#Gray
		water_color = [0.,0.,0.]		#Black
		shadow_color = [1.0, .545, .0]  #Orange
		cloud_color = [.27, .15, .50]		#Purple

		custom_colormap = [empty_color,snow_color,gray_color,pond_color,water_color,shadow_color,cloud_color]
		custom_colormap = colors.ListedColormap(custom_colormap)

		#Making sure there is atleast one of every pixel so the colors map properly (only changes
		# display image, not saved data)
		image[0][0] = 0
		image[1][0] = 1
		image[2][0] = 2
		image[3][0] = 3
		image[4][0] = 4
		image[5][0] = 5
		image[6][0] = 6

	mimg.imsave(save_name, image, format='png', cmap=custom_colormap)

#### Count the number of pixels in each classification category of given image
def count_features(classified_image):

	sum_snow = float(len(classified_image[classified_image==1.0]))
	sum_gray_ice = float(len(classified_image[classified_image==2.0]))
	sum_melt_ponds = float(len(classified_image[classified_image==3.0]))
	sum_open_water = float(len(classified_image[classified_image==4.0]))
	sum_shadow = float(len(classified_image[classified_image==5.0]))

	# num_pixels =  sum_snow + sum_gray_ice + sum_melt_ponds + sum_open_water

	return sum_snow, sum_gray_ice, sum_melt_ponds, sum_open_water, sum_shadow

# Code from http://chriskiehl.com/article/parallelism-in-one-line/
# Returns a list of .h5 files in the given folder.
def get_image_paths(folder,keyword='.h5'):
	return (os.path.join(folder, f)
		for f in os.listdir(folder)
		if keyword in f)

# Remove hidden folders and files from the given list of strings (mac)
def remove_hidden(folder):
	i = 0
	while i < len(folder):
		if folder[i][0] == '.':
			folder.pop(i)
		else:
			i+=1

# Plots a confusion matrix. Adapted from 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
# 
def plot_confusion_matrix(cm,categories,ylabel,xlabel,
							normalize=False,
							title='',
							cmap=plt.cm.Blues):
	"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	font = {'family' : 'Times New Roman',
		'weight' : 'bold',
		'size'   : 12}

	matplotlib.rc('font', **font)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	tick_marks = np.arange(len(categories))
	plt.xticks(tick_marks, categories, rotation=45)
	plt.yticks(tick_marks, categories)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
	    print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 4.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.show()
