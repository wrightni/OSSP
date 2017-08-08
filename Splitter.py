## Method for splitting a large image into more manageable chunks. 
#1gb output chunks seems to be a good filesize (ie 16gb image -> 16 x 1gb image)

import argparse
import time
import os
import math
import h5py
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import gdal
from skimage import exposure


def split_image(input_path, image_name, image_type, output_path=None, number_of_splits=9, verbose=False):
	'''
	Reads an image file and splits it into a grid of square segments.
	The number of segments is the nearest perfect square to number_of_splits.
	Split images are saved as .h5 formatted datasets
	'''
	full_image_name = os.path.join(input_path, image_name)
	# Check to make sure the given file exists
	if os.path.isfile(full_image_name):
		if verbose: print "Reading image..."
		start_time = time.clock() 
		dataset = gdal.Open(full_image_name)
	else:
		print "File not found: " + image_name
		return None

	# If no output directory was provided, default to a /split directory in
	# 	the input directory
	if output_path == None:
		output_path = os.path.join(input_path,"splits")
	# If the output path doesnt already exist, create that directory
	if not os.path.isdir(output_path):
		os.makedirs(output_path)

	# Round number_of_splits to the nearest square number
	base = round(math.sqrt(number_of_splits))
	number_of_splits = base*base

	# Approximate size of subimages created in Watershed.py.
	subimage_size = 500	

	# Pull the image date from the header information, and prompt user for manual entry if 
	# the date field does not exist in header. 
	metadata = dataset.GetMetadata()
	try:
		if image_type == 'srgb':
			header_date = metadata['EXIF_DateTime']
			image_date = header_date[5:7] + header_date[8:10]
		elif image_type == 'pan' or image_type == 'wv02_ms':
			image_date = metadata['NITF_STDIDC_ACQUISITION_DATE'][4:8]
		image_date = int(image_date)
	except:
		image_date = raw_input("Could not find image date, please enter in mmdd: ")
		image_date = int(image_date)

	if verbose: print "Using %i as image date." %image_date

	# Make sure each split is big enough for at least 2 subimages in each dimension. If not, there's
	# really little reason to be splitting this image in the first place. 
	while dataset.RasterXSize/base < subimage_size*2 or dataset.RasterYSize/base < subimage_size*2:
		base -= 1
		if verbose: print "Too many splits chosen, reducing to: %i" %(base*base)
	
	# Creates dimensions for a square based on the number of splits requested.
	# Integer truncation rounds down to the nearest whole number
	WIDTH = int(base)
	HEIGHT = WIDTH
	BANDS = dataset.RasterCount

	# Define the pixel dimension of each split with integer truncation
	# Round to the nearest multiple of 100. This makes creating subimages used for watershed segmenation
	#	easier, and limits the number of pixels in each dimension that will be truncated to 100 max
	
	divisor = 100*WIDTH
	cols = int(dataset.RasterXSize / divisor)*divisor
	rows = int(dataset.RasterYSize / divisor)*divisor
	if verbose: print "Image Dimensions: %i x %i" %(cols, rows)
	y_size = int(cols / WIDTH)
	x_size = int(rows / HEIGHT)
	if verbose: print "Split Dimensions: %i x %i" %(y_size, x_size)
	if verbose: print "Number of Bands: %i" %BANDS

	#### Splits the image into number_of_split parts
	# Saves each segment in its own .h5 file with n datasets; one for each spectral band.
	
	# For sRGB:
	# Set the percentile thresholds at a temporary value until finding the apropriate ones
	# considering all three bands. 
	if image_type == 'srgb':
		p01 = -1
		p98 = -1
		# Create a queue to hold all of the bands so we only have to read the dataset once
		all_bands = deque([])

	for b in range(1,BANDS+1):

		if verbose: print "Reading band %s data..." %b

		# Read the band information from the gdal dataset
		band = dataset.GetRasterBand(b)
		# [band_min, band_max, band_mean, band_std] = band.ComputeStatistics(False)
		band = np.array(band.ReadAsArray())

		# Calculate the range of pixel values that we want to keep in the final image.
		#	We discared the bottom .01% and top 2% of pixel intensity values (percent of total number
		#	of pixels). This tends to stretch brighter pixels more towards the high end. This is OK
		#	because all bright pixels fall will in the ice category anyways. The bottom .01% threshold
		#	was chosen to remove tails from the intensity histogram, but should maintain
		#	the spectral signature of water and melt ponds.
		# Find the percentiles for this band, and update the 
		# Only check band values that are greater than 1, ignoring the empty pixels with value 0.

		p01_b = 0
		p98_b = 0

		# Criteria for a peak:
		# Distance to the nearest neighbor is greater than one third the approx. dynamic range
		# Has a minimum number of pixel counts (based on image size, loosely)
		# Is greater than its immediate neighbors, and neighbors 5 bins to either side

		hist, bin_centers = exposure.histogram(band)
		if image_type == 'srgb':
			threshold = 1000
		else:
			threshold = 100000
		local_peaks = []
		for i in range(1,len(bin_centers)-1):
			#Acceptable width of peak is +/-5, except in edge cases
			if i < 5:
				w_l = i
				w_u = 5
			elif i > len(bin_centers)-6:
				w_l = 5
				w_u = len(bin_centers)-1-i
			else:
				w_l = 5
				w_u = 5
			if hist[i] >= hist[i+1] and hist[i] >= hist[i-1] and hist[i] >= hist[i-w_l] and hist[i] >= hist[i+w_u]:
				if hist[i]>threshold:
					local_peaks.append(bin_centers[i])

		num_peaks = len(local_peaks)
		distance = 5		#Initial distance threshold
		distance_threshold = int((local_peaks[-1]-local_peaks[0]) / 6)	# One third the 'dynamic range' (radius)
		# Min threshold
		if distance_threshold<=5: 
			distance_threshold=5
		# Looking for three main peaks corresponding to the main surface types: open water, MP and snow/ice
		# But any peak that passes the criteria is fine.
		while distance <= distance_threshold:
			i = 0
			to_remove = []
			# Cycle through all of the peaks
			while i < num_peaks-1:
				# Check the current peak against the adjacent one. If they are closer than
				#  the threshold distance, delete the lower valued peak
				if local_peaks[i+1]-local_peaks[i]<distance:
					if hist[np.where(bin_centers==local_peaks[i])[0][0]] < hist[np.where(bin_centers==local_peaks[i+1])[0][0]]:
						to_remove.append(local_peaks[i])
					else:
						to_remove.append(local_peaks[i+1])
						i+=1	#Add an extra to the index, because we don't need to check the next peak again
				i+=1

			# Remove all of the peaks that did not meet the criteria above
			for j in to_remove:
				local_peaks.remove(j)

			# Recalculate the number of peaks left, and increase the distance threshold
			num_peaks = len(local_peaks)
			distance+=5

		# Find the indicies of the highest and lowest peak, by intensity (not # of pixels)
		# Then search for an intensity threshold that is greater than the higher peak and 
		# less than 2% of the number of pixels, and one that is lower than the lowest peak and
		# 30% of the number of pixels. These will be our stretching thresholds. 
		# 2% and 30% picked to give similar results to what p01 and p98 used to find.  
		max_peak = np.where(bin_centers==local_peaks[-1])[0][0]	#Max intensity that is
		thresh_top = max_peak
		while hist[thresh_top] > hist[max_peak]*0.1: #1?
			thresh_top+=2

		min_peak = np.where(bin_centers==local_peaks[0])[0][0]	#Max intensity that is
		thresh_bot = min_peak
		while hist[thresh_bot] > hist[min_peak]*0.5:
			thresh_bot-=1

		# Assign these the p01 and p98 (legacy variable names)
		p01_b = bin_centers[thresh_bot]
		p98_b = bin_centers[thresh_top]

		# Limit the amount of stretch to a percentage of the total dynamic range
		# 8 bit vs 11 bit (WorldView)
		# 256   or 2048
		# We can identify images that should have the stretch limited by their standard deviation.
		# Images that do not have all three classes (Open water and Ice, most importantly) will have
		# much lower stdev than images containing all three. 
		if image_type == 'pan' or image_type == 'wv02_ms':
			max_bit = 2047	#While the images are 11bit, white ice tends to be ~600-800 intensity
			upper_limit = 0.25
		else:
			max_bit = 255
			upper_limit = 0.6
		min_range = int(max_bit*.08)
		max_range = int(max_bit*upper_limit)

		if verbose: print "Stretching: %i to 0 and : %i to 255" %(p01_b,p98_b)

		if verbose: print "min range: {0:0f} | max range: {1:0f}".format(min_range,max_range)

		if num_peaks < 3:
			if p01_b > min_range:
				p01_b = min_range
			if p98_b < max_range:
				p98_b = max_range
		## _______

		# if verbose: print "Stretching: %i to 0 and : %i to 255" %(p01_b,p98_b)

		# In standard rgb imagery we want to scale each band by the min and max
		# of all bands. Here we check the current p01 and p98 values against the
		# lowest and highest previously found, and adjust if necessary.
		if image_type == 'srgb':
			if p01_b < p01 or p01 == -1:
				p01 = p01_b
			if p98_b > p98 or p98 == -1:
				p98 = p98_b
			all_bands.append(band)
		# In WV imagery we want to scale each band independently, based on its own
		# min and max values. Therefore we don't need to hold values, and can proceed
		# to stretching and saving the current band directly. 
		else:
			if verbose: print "Splitting band %s..." %b
			success = split_and_save(band, b, image_type, image_date, WIDTH, HEIGHT, x_size, y_size, 
				image_name, output_path, p01_b, p98_b, verbose)
			# If the split and save function determined that the split already exists, we can exit here.
			if success == False:
				return

			if verbose: print "Band %s complete" %b
			band = None

	# Close the gdal dataset
	dataset = None

	# Flag for downsampling images in the split
	downsample = False

	factors = [1]

	for f in factors:

		# Now that we've checked the histograms of each band in the srgb image, we can rescale and save each band.
		if image_type == 'srgb':
			for b in range(1,BANDS+1):
				# Remove bands from the queue as they are used to free up memory space.
				# band = all_bands.popleft()
				band = all_bands[b-1]
				if downsample == True:
					band = downsample(band,f)
					image_name = os.path.splitext(image_name)[0] + "_{0:01d}f".format(f) + os.path.splitext(image_name)[1]

				if verbose: print "Splitting band %s..." %b
				success = split_and_save(band, b, image_type, image_date, WIDTH, HEIGHT, x_size, y_size, 
					image_name, output_path, p01, p98, verbose)
				# If the split and save function determined that the split already exists, we can exit here.
				if success == False:
					return

				if verbose: print "Band %s complete" %b

	elapsed_time = time.clock() - start_time
	if verbose: print "Done. "
	if verbose: print "Time elapsed: {0}".format(elapsed_time)

# 'Downsample' an image by the given factor. Every pixel in the resulting image
# is the result of an average of the NxN kernel centered at that pixel, where N
# is factor.
def downsample(band,factor):
	# from scipy.ndimage.filters import convolve
	from skimage.measure import block_reduce

	band_downsample = block_reduce(band, block_size=(factor,factor),func=np.mean)
	band_copy = np.zeros(np.shape(band))
	for i in range(np.shape(band_downsample)[0]):
		for j in range(np.shape(band_downsample)[1]):
			band_copy[i*factor:(i*factor)+factor,j*factor:j*factor+factor] = band_downsample[i,j]

	return band_copy


#### Rescale and image data from range [bottom,top] to uint8 ([0,255])
def rescale_band(band, bottom, top):

	# Record pixels that contain no spectral information, indicated by a value of 0
	empty_pixels = np.zeros(np.shape(band),dtype='bool')
	empty_pixels[band == 0] = True

	# Rescale the data to use the full int8 (0,255) pixel value range.
	# Check the band where the values of the matrix are greater than zero so that the
	# percentages ignore empty pixels.
	stretched_band = exposure.rescale_intensity(band, in_range=(bottom,top), out_range=(1,255))
	new_band = np.array(stretched_band,dtype=np.uint8)
	# Set the empty pixel areas back to a value of 0. 
	new_band[empty_pixels] = 0

	return new_band

#### Split an image band and save it to an hdf5 file.
def split_and_save(band, band_num, image_type, image_date,
	WIDTH, HEIGHT, x_size, y_size, image_name, output_path, bottom, top, verbose):

	num = 0
	num_splits = WIDTH*HEIGHT
	for i in range(WIDTH):
		for j in range(HEIGHT):
			num += 1
			output_name = os.path.splitext(image_name)[0] + "_s{0:02d}of{1:02d}.h5".format(num,num_splits)
			
			output_exists = os.path.isfile(os.path.join(output_path, output_name))
			#On the first iteration, check to see if the file already exists. If it does, break the loop and don't overwrite
			# Return false so the rest of the program knows not to continue
			if band_num == 1:
				if output_exists:
					print "Split already exists"
					return False

			# Rescale the image one block at a time
			if verbose: print "Rescaling band %i block %i..." %(band_num,num)
			band_block = band[i*x_size:(i+1)*x_size, j*y_size:(j+1)*y_size]
			rescale_block = rescale_band(band_block,bottom,top)

			# Want to keep all 8 bands of each split in the same file. This creates the file if it does not already exist, 
			#	and adds to the file if it does. Might be a better way to do this?

			if output_exists:
				outfile = h5py.File(os.path.join(output_path, output_name), "r+")
			else:
				outfile = h5py.File(os.path.join(output_path, output_name), "w")
				outfile.attrs.create("Image Type",image_type)
				outfile.attrs.create("Image Date",image_date)
			outfile.create_dataset('Band_' + str(band_num), data=rescale_block, dtype='uint8', compression='gzip')
			outfile.close()

	# Return True if this split saved successfully (i.e. the split does not already exist)
	return True

def main():
	
	#### Set Up Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("input_dir", 
						help="directory path for input image")
	parser.add_argument("filename",
						help="name of image")
	parser.add_argument("image_type", type=str, choices=['srgb','wv02_ms','pan'],
						help="image type: 'srgb', 'wv02_ms', 'pan'")
	parser.add_argument("--output_dir", metavar="dir",
						help="directory path for output images")
	parser.add_argument("-s", "--splits", type=int, default=9, metavar="int",
						help='''number of splits to perform on the input images. This is rounded 
						to the nearest perfect square''')
	parser.add_argument("-v", "--verbose", action="store_true",
						help="display status updates")

	#### Parse Arguments
	args = parser.parse_args()
	input_path = os.path.abspath(args.input_dir)
	image_name = args.filename
	image_type = args.image_type
	if args.output_dir:
		output_path = os.path.abspath(args.output_dir)
	else:
		output_path = None
	number_of_splits = args.splits
	verbose = args.verbose
	
	#### Split Image with Given Arguments
	split_image(input_path, image_name, image_type, output_path, number_of_splits, verbose)

if __name__ == "__main__":
	main()

