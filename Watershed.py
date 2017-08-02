#title: Watershed Transform
#author: Nick Wright
#adapted from: Justin Chen, Arnold Song

import argparse
import time
import numpy as np
import os
import h5py
import warnings
import math

import skimage
from skimage import filters, morphology, feature, exposure, segmentation, future
from scipy import ndimage
from scipy.misc import bytescale
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.colors as colors
import matplotlib.image as mimg
from lib import utils


class Subimage:
	'''
	An object for an MxN image segment. M and N are defined in read_image()
	Has the following properties:
		main_band: spectral band 1
		alternate_bands: bands 2-8 (a dictionary of {band_id:band_data}
		watershed: The watershed transformation of band 1
		empty_pixels: boolean matrix where True corresponds to pixels with no spectral data
		
	'''
	#Initialize the data to be stored in this object
	def __init__(self, data):
		self.main_band = data
		self.watershed = None
		self.sobel_image = np.zeros(np.shape(data))
		self.alternate_bands = {}
		# self.rag = None 					#Region Adjacency Graph
		
		#Checking to see if the subimage has any data
		sum_pixels = np.sum(data)
		if sum_pixels == 0:
			self.valid_subimage = False
		else:
			self.valid_subimage = True

		#Stores the locations that contain no spectral data
		self.empty_pixels = np.zeros(np.shape(data),dtype='bool')
		self.empty_pixels[data == 0] = True

	#Method for adding bands 2-8 to the object, stored in a dictionary
	def add_band(self, band_id, band_data):
		if self.valid_subimage == True:
			self.alternate_bands[band_id] = band_data	
		
	def get_main_band(self):
		return self.main_band
		
	def get_alternate_band(self, band_id):
		if self.valid_subimage == False:
			return np.zeros(np.shape(self.main_band))

		if band_id == 1:
			return self.main_band
		elif band_id in self.alternate_bands:
			return self.alternate_bands[band_id]
		else:
			return self.main_band
	
	def get_watershed(self):
		return self.watershed

	#Runs a watershed transform on the main dataset
	#	1. Create a gradient image based on the sobel method
	#	2. Transform the gradient image into a binary image using a threshold, such as otsu
	#		-Finds all areas in the image that have a gradient larger than our threshold
	#	3. Find the local maximum distances to the nearest area with a significant gradient and place a marker
	#	4. Build a watershed image from the sobel image using the markers
	def watershed_transformation(self, sobel_threshold, amplification_factor):
		if self.valid_subimage == True:

			#Create a gradient image using a sobel filter
			self.sobel_image = filters.sobel(self.main_band)

			upper_threshold = np.amax(self.sobel_image)/amplification_factor
			if upper_threshold < 0.20:
				upper_threshold = 0.20
			self.sobel_image = exposure.rescale_intensity(self.sobel_image, in_range=(0,upper_threshold), out_range=(0,1))

			# gauss_im = ndimage.gaussian_filter(self.main_band,2)
			# edge_im = feature.canny(self.main_band, sigma=1)

			# Prevents the watersheds from 'leaking' along the sides of the image
			self.sobel_image[:,0] = 1
			self.sobel_image[:,-1] = 1
			self.sobel_image[0,:] = 1
			self.sobel_image[-1,:] = 1

			# Set all values in the sobel image that are lower than the 
			# given threshold to zero. 
			self.sobel_image[self.sobel_image<=sobel_threshold]=0

			# Find local minimum values in the sobel image by inverting
			# sobel_image and finding the local maximum values
			inv_sobel = 1-self.sobel_image
			local_min = feature.peak_local_max(inv_sobel,min_distance=3, indices=False)
			markers = ndimage.label(local_min)[0]

			# Build a watershed from the markers on top of the edge image
			self.watershed = morphology.watershed(self.sobel_image,markers)
			self.watershed = np.array(self.watershed,dtype='uint16')

			# Set all values outside of the image area (empty pixels, usually caused by
			# orthorectification) to one value, at the end of the watershed list.
			self.watershed[self.empty_pixels]=np.amax(self.watershed)+1

		else:
			# print ":("
			self.watershed = np.zeros(np.shape(self.main_band),dtype='uint16')
		
	#Plots a watershed image on top of and beside the original image
	def display_watershed(self):

		ws_bound = segmentation.find_boundaries(self.watershed)

		ws_display = create_composite(self.main_band,self.main_band,self.main_band)
		ws_display[:,:,0][ws_bound] = 98
		ws_display[:,:,1][ws_bound] = 202
		ws_display[:,:,2][ws_bound] = 202

		display_im = create_composite(self.main_band,self.get_alternate_band(1),self.get_alternate_band(2))

		fig, axes = plt.subplots(1,3,subplot_kw={'xticks':[], 'yticks':[]})
		fig.subplots_adjust(hspace=0.3,wspace=0.05)

		axes[1].imshow(self.sobel_image,interpolation='none')
		axes[0].imshow(display_im,interpolation='none')
		# axes[2].imshow(ws_display,interpolation='none')

		# randcolor = colors.ListedColormap(np.random.rand(256,3))

		axes[2].imshow(ws_display,interpolation='none')
		plt.show()

		# aa = raw_input("Save? ")
		# # aa = '582'
		# if aa == 'n':
		# 	return
		# else:
		# 	save_number = int(aa)

		# 	segim = self.watershed
		# 	segim[segim!=save_number]=0
		# 	ws_bound2 = segmentation.find_boundaries(segim)
		# 	ws_bound2 = morphology.binary_dilation(ws_bound2)
		# 	saveim = create_composite(self.main_band,self.get_alternate_band(1),self.get_alternate_band(2))
		# 	saveim[:,:,0][ws_bound2] = 255
		# 	saveim[:,:,1][ws_bound2] = 22
		# 	saveim[:,:,2][ws_bound2] = 22

		# 	# saveim = saveim[:500,:500,:]

		# 	savepath = '/Users/nicholas/Documents/Dartmouth/Projects/Methods Paper/figures/segment.png'
			
		# 	mimg.imsave(savepath,saveim)


		return


#Read in Image and creates the subimage objects with all 8 bands of data.
#	Adds the subimage objects to a list that represents the whole image.
#	Note: Only bands 1,5,3,2 are used, though all 8 are added to the subimage object
#		(1 for the transformation, and 5,3,2 to create a RBG image)
def read_image(full_file_name, verbose):
	'''
	Reads filepath/image_name
	Returns list of subimage objects with 8 bands
	 		the complete band 1 (rescaled), 
			the number of subimages in each direction
			the number of bands.
	'''

	#A check to make sure the input filepath and file exist
	if os.path.isfile(full_file_name):
		input_file = h5py.File(full_file_name, 'r')
		image_type = input_file.attrs.get("Image Type")
		image_date = input_file.attrs.get("Image Date")
		im_attributes = [image_type,image_date]
		if verbose: 
			start_time = time.clock()
			print "Reading file..."
	else:
		print full_file_name
		print "File not found."
		return None, None, None, None, None, None
	
	# The inputfile has 1 dataset for every band (see Splitter.py)
	BANDS = len(input_file.keys())
	
	#In the specifc case of an 8 band WV image, we want the main band to be the red band
	# This band has a good amount of distinction between water and ice intensities
	#  main band is the one used for segmentation
	if BANDS == 8:
		main_band = input_file['Band_5'][:]
	else:
		main_band = input_file['Band_1'][:]

	BLOCK_SIZE_X, BLOCK_SIZE_Y = find_blocksize(main_band.shape[1],main_band.shape[0])

	num_x_subimages = int(main_band.shape[1] / BLOCK_SIZE_X) #integer truncation intentional
	num_y_subimages = int(main_band.shape[0] / BLOCK_SIZE_Y)

	if verbose: 
		print "Number of x subimages: %s" %num_x_subimages
		print "Number of y subimages: %s" %num_y_subimages


	# Skip this file if the entire image is empty. Return the empty band to save
	#	for continuity sake. 
	if np.sum(main_band) == 0:
		print "Empty split"
		input_file.close()
		return None, main_band, num_x_subimages, num_y_subimages, None, im_attributes
	
	#This will be a list of all the objects created by the subimage class
	subimage_list = []

	# These don't include pixels left over after integer truncation, so I changed
	# the splitter function to save all splits in amounts divisible by 500 to get around this for now. 
	for i in range(num_y_subimages):
		for j in range(num_x_subimages):
			subimage_list.append(Subimage(main_band[i*BLOCK_SIZE_Y:(i+1)*BLOCK_SIZE_Y, j*BLOCK_SIZE_X:(j+1)*BLOCK_SIZE_X]))

	#This adds the remaining spectral bands to the subimage object, and rescales / stretches
	# the data in the same way as seen above with main_band
	if BANDS != 1:
		for band_id in range(1,BANDS+1):
			band_data = input_file['Band_' + str(band_id)][:]
			# For WV 8 band MS, we already added band 5
			if BANDS == 8 and band_id == 5:
				continue
 			subimage_counter = 0
			for i in range(num_y_subimages):
				for j in range(num_x_subimages):
					subimage_list[subimage_counter].add_band(band_id,
						band_data[i*BLOCK_SIZE_Y:(i+1)*BLOCK_SIZE_Y, j*BLOCK_SIZE_X:(j+1)*BLOCK_SIZE_X])
					subimage_counter += 1

	#Close the input dataset and clearing memory
	input_file.close()
	empty_pixels = None

	if verbose: 
		elapsed_time = time.clock() - start_time	
		print "Done. "
		print "Time elapsed: {0}".format(elapsed_time)

	return subimage_list, main_band, num_x_subimages, num_y_subimages, BANDS, im_attributes

# Finds the appropriate block size for an input image of given dimensions
# Method returns the first factor of the input dimension that is greater than n
def find_blocksize(x_dim, y_dim):
	
	desired_size = 600
	# Just in case x_dim and y_dim are smaller than expected
	if x_dim < desired_size or y_dim < desired_size:
		block_x = x_dim
		block_x = y_dim

	factors_x = factor(x_dim)
	factors_y = factor(y_dim)
	for x in factors_x:
		if x >= desired_size:
			block_x = x
			break
	for y in factors_y:
		if y >= desired_size:
			block_y = y
			break

	return int(block_x), int(block_y)

# Returns a sorted list of all of the factors of number
# using trial division.
#	http://www.calculatorsoup.com/calculators/math/factors.php
def factor(number):

	factors = []
	s = int(math.ceil(math.sqrt(number)))

	for i in range(1,s):
		c = float(number) / i
		if int(c) == c:
			factors.append(c)
			factors.append(number/c)

	factors.sort()
	
	return factors

# Combines multiple bands (RBG) into one 3D array
# Adapted from:  http://gis.stackexchange.com/questions/120951/merging-multiple-16-bit-image-bands-to-create-a-true-color-tiff
# Useful band combinations: http://c-agg.org/cm_vault/files/docs/WorldView_band_combs__2_.pdf
def create_composite(red, green, blue):
	img_dim = red.shape
	img = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.uint8)
	img[:,:,0] = red
	img[:,:,1] = green
	img[:,:,2] = blue
	return img


#get_distance and get_markers are methods used in the watershed transformation to create
#	the seed points based on the binary Otsu image. 
def get_distance(image):
	distance_image = ndimage.distance_transform_edt(image)
	shape = np.shape(distance_image)
	extenshape = (shape[0]+20,shape[1]+20)
	extendist = np.zeros(extenshape)
	extendist[10:shape[0]+10,10:shape[1]+10]=distance_image

	return extendist, shape

def get_markers(extendist, shape):
	extenlocal_max = feature.peak_local_max(extendist,min_distance=3, threshold_rel=0., indices=False)
	local_max = extenlocal_max[10:shape[0]+10,10:shape[1]+10]
	markers = ndimage.label(local_max)[0]

	return markers

def process(file_path, image_name, output_path=None, hist_check=False, color_check=False, test_check=False, verbose=False):
	
	if output_path == None:
		output_path = os.path.join(file_path,"processed")

	try: 
		os.makedirs(output_path)
	except OSError:
		if not os.path.isdir(output_path):
			raise

	full_file_name = os.path.join(file_path, image_name)
	output_filename = os.path.splitext(image_name)[0] + '_segmented.h5'
	
	#Calls the read_image method, returning:
	#	subimage_list: array of all subimage objects
	#	full_band: the complete (nonsegmented) band 1 data that has been rescaled and inverted
	#	number of subimages in the x and y directions, and the number of bands read.
	subimage_list, full_band, num_x_subimages, num_y_subimages, number_of_bands, im_attributes = read_image(full_file_name, verbose)

	#Unpack attributes
	image_type = im_attributes[0]
	image_date = im_attributes[1]

	if subimage_list == None and np.sum(full_band) == 0:
		if verbose: print "Saving output files..."
		# Save the watershed with a placeholder
		watershed_segments = np.zeros(np.shape(full_band))
		outfile = h5py.File(os.path.join(output_path, output_filename), "w")
		outfile.attrs.create("Image Type", image_type)
		outfile.attrs.create("Image Date", image_date)
		outfile.create_dataset('watershed', data=watershed_segments,compression='gzip',compression_opts=9)
		outfile.create_dataset('dimensions', data=[num_x_subimages,num_x_subimages])
		outfile.create_dataset('original_1', data=full_band, compression='gzip', compression_opts=9)
		outfile.close()
		return
	elif subimage_list == None:
		return

	#A method for visually inspecting the threshold results in histogram form.
	if hist_check == True:
		hist2 = np.zeros((255,))
		hist = np.zeros((255,))

		hist2, bin_centers2 = exposure.histogram(full_band[full_band>=1])

		plt.figure(1)
		plt.bar(bin_centers2, hist2)
		# p01,p98 = np.percentile(main_band[main_band>=1], (0.01,98))

		# plt.plot([p01,p01],[0,np.max(hist2[0:np.max(full_band)])])
		# plt.plot([p98,p98],[0,np.max(hist2[0:np.max(full_band)])])
		plt.xlim((0,np.max(full_band)))
		plt.ylim((0,np.max(hist2[0:np.max(full_band)])))
		plt.xlabel("Pixel Intensity")
		plt.ylabel("Frequency")
		plt.show()

	if verbose: 
		start_time = time.clock()
		prog1 = 0
		prog2 = 10

	# Set the amp factor and lower threshold based on the input image

	if image_type =='pan' or 'wv02_ms':	
		sobel_threshold = 0.1
		amplification_factor = 2.
	if image_type =='srgb':	
		sobel_threshold = 0.1
		amplification_factor = 3

	# Watershed segments is a variable that stores the result of the watershed
	# segmentation, where each element of the array is one subimage. 
	watershed_segments = []
	all_band_segments = {} #{band_id: [subimage][row][column]}

	for subimage in subimage_list:
		
		subimage.watershed_transformation(sobel_threshold,amplification_factor)
		watershed_segments.append(subimage.get_watershed())
		
		#Concurrently compiles all of the data from each band that will be 
		#exported along side the watershed transformation. 
		for band_id in range(1,number_of_bands+1):
			if band_id in all_band_segments:
				all_band_segments[band_id].append(subimage.get_alternate_band(band_id))
			else:
				all_band_segments[band_id] = [subimage.get_alternate_band(band_id)]
		
		# print np.amin(subimage.main_band), np.amax(subimage.m)

		if verbose: 
			#A counter to display the status to the user. 
			if int(float(prog1)/float(len(subimage_list))*100) == prog2:
				print "%s Percent" %prog2
				prog2 += 10
			prog1 += 1
			# if prog1 == 25: quit()

	if verbose: 
		print "100 Percent"
		elapsed_time = time.clock() - start_time
		print "Time elapsed: {0}".format(elapsed_time)

	watershed_segments = np.array(watershed_segments)
	
	#Option to display a watershed image a long side a raw image for the user
	#to see the result of the transformation.
	while test_check == True:
		choice = raw_input("Display subimage/wsimage pair? (y/n): ")
		if choice == 'y':
			selection = int(raw_input("Choose image (0," +str(len(subimage_list)) + "): "))
			if selection >= 0 and selection < len(subimage_list):
				subimage = subimage_list[selection]
				# color_composite = create_composite(subimage.get_alternate_band(5), subimage.get_alternate_band(3), subimage.get_alternate_band(2))
				
				subimage_list[selection].display_watershed()
			else:
				print "Invalid subimage index." 
		else:
			test_check = False
		if test_check == False:
			save_flag = raw_input("Save results? (y/n): ")
			if save_flag == 'n':
				quit()

	## Saving the watershed results and raw image data together
	if verbose: print "Saving output files..."
	outfile = h5py.File(os.path.join(output_path, output_filename), "w")
	outfile.attrs.create("Image Type", image_type)
	outfile.attrs.create("Image Date", image_date)
	outfile.create_dataset('watershed', data=watershed_segments,compression='gzip',compression_opts=3)
	outfile.create_dataset('dimensions', data=[num_x_subimages,num_y_subimages])
	for band_id in all_band_segments.keys():
		band = np.array(all_band_segments[band_id])
		outfile.create_dataset('original_' + str(band_id), data=band, compression='gzip', compression_opts=3)
	
	outfile.close()

	# Saves an rbg png if the color check arg was used
	if color_check == True:

		if verbose: print "Saving color image..."
		holder = []
		# Find the appropriate bands to use for an rgb representation
		if image_type == 'wv02_ms':
			rgb = [5,3,2]
		elif image_type == 'srgb':
			rgb = [1,2,3]
		else:
			rgb = [1,1,1]

		red_band = all_band_segments[rgb[0]]
		green_band = all_band_segments[rgb[1]]
		blue_band = all_band_segments[rgb[2]]

		for i in range(len(red_band)):
			holder.append(create_composite(
				red_band[i], green_band[i], blue_band[i]))

		colorfullimg = utils.compile_subimages(holder, num_x_subimages, num_y_subimages, 3)
		
		# colorfullimg = create_composite(holder[0][0,:,:], holder[1][0,:,:], holder[2][0,:,:])
		savepath = os.path.join(output_path, os.path.splitext(image_name)[0] + '_RGB.png')
		mimg.imsave(savepath,colorfullimg)
		colorfullimg = None
		if verbose: print "Done."


	# Remove the image datasets from memory
	all_band_segments = None
	watershed_segments = None
	subimage_list = None
	full_band = None

	if verbose: print "Done."


def main():
	
	#### Set Up Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("input_dir", 
						help="directory containing a folder of image splits in .h5 format")
	parser.add_argument("filename",
						help="name of image split")
	parser.add_argument("--output_dir", metavar="dir",
						help="directory path for output images")
	parser.add_argument("--histogram", action="store_true",
						help="display histogram of pixel intensity values before segmentation")
	parser.add_argument("-c", "--color", action="store_true",
						help="save a color (rgb) image of the input image")
	parser.add_argument("-t", "--test", action="store_true",
						help="inspect results before saving")
	parser.add_argument("-v", "--verbose", action="store_true",
						help="display text information and progress")

	#### Parse Arguments
	args = parser.parse_args()
	input_path = os.path.abspath(args.input_dir)
	split_path = os.path.join(input_path)
	image_name = args.filename
	if args.output_dir:
		output_path = os.path.abspath(args.output_dir)
	else:
		output_path = None
	hist_check = args.histogram
	color_check = args.color
	test_check = args.test
	verbose = args.verbose

	process(split_path, image_name, output_path, hist_check, color_check, test_check, verbose)

if __name__ == "__main__":
	main()
