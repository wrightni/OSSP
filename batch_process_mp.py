#Batch Process
#Usage: Fully processes all images in the given directory with the given training data. 

import os
import shutil
import argparse
import multiprocessing
import time
import random
import itertools
import h5py

from Splitter import split_image
from Watershed import process as segment_image
from RandomForest import process as classify_image

from lib import utils


def main():

	#### Set Up Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("input_dir", 
						help="directory path containing date directories of images to be processed")
	parser.add_argument("image_type", type=str, choices=['srgb','wv02_ms','pan'],
						help="image type: 'srgb', 'wv02_ms', 'pan'")
	parser.add_argument("training_dataset",
						help="training data file")
	parser.add_argument("--training_label", type=str, default=None,
						help="name of training classification list")
	parser.add_argument("-s", "--splits", metavar='int', type=int, default=9,
						help="number of subdividing splits to preform on raw image")
	parser.add_argument("-p", "--parallel", metavar='int', type=int, default=1,
						help="number of parallel processes to run. It is typically prudent to leave at least 1 core unused. Beware of memory usage requirements for each process.")

	#### Parse Arguements
	args = parser.parse_args()
	#System filepath that contains the directories that we will work for batch processing
	root_dir = args.input_dir
	image_type = args.image_type
	tds_file = args.training_dataset
	if args.training_label is None:
		tds_list = image_type
	else:
		tds_list = args.training_label
	num_splits = args.splits
	num_cores = args.parallel

	# Make sure the user doesn't try to use more cores than they have. 
	if num_cores > multiprocessing.cpu_count():
		num_cores = multiprocessing.cpu_count()-1

	#### Process images in the given directory.
	#		Expected filestructure: all images in date specific folder

	for path, directories, files in os.walk(root_dir):

		utils.remove_hidden(directories)
		utils.remove_hidden(files)
		
		#### Split Each .tif

		# Loop through contents in the date directory
		for file in files:

			# Find .tif or .jpg formatted files
			if os.path.splitext(file)[1].lower() == '.tif' or os.path.splitext(file)[1].lower() == '.jpg':
				# We know we're working with an image now
				image_name = file
				# split_flag, true if we are going to split this image, false if it has already been split
				split_flag = True
				
				split_dir = os.path.join(path,'splits')
				# If theres no split directory, we know that we need to proceed with the split
				if os.path.isdir(split_dir) is False:
					os.makedirs(split_dir)
				else:	
					# Check to see if the image has already been split by looking in the 'splits' subdirectory
					splits = next(os.walk(split_dir))[2]	#list of files in split directory
					# Loop through all files in the in splits directory; find any that match the current image
					# Potentially revisit this to make it more robust (completed flag might trigger on files other than splits)
					for split in splits:
						if os.path.splitext(image_name)[0] in split:
							split_flag = False
					# Check the processed directory, if the split has already been classified, we don't need to
					#	split it again
					if os.path.isdir(os.path.join(split_dir,'processed')):
						completed_files = os.listdir(os.path.join(split_dir,'processed'))
						for completed_name in completed_files:
							if os.path.splitext(image_name)[0] in completed_name:
								split_flag = False

				# If the current image has not already been split, call the split function.
				if split_flag is True:
					print "Splitting image: %s" %os.path.join(path,image_name)
					print "-"*80
					split_image(path, image_name, image_type, number_of_splits=num_splits)
				else:
					print "Already Split: %s" %os.path.join(path,image_name)

		# If there is a directory with splits, continue to process
		if os.path.isdir(os.path.join(path,'splits')):

			#### Load Training Data
			tds = utils.load_tds(tds_file,tds_list)
		
			#### Analyze Splits

			# Find the list of splits that we need to process
			splits = get_image_paths(os.path.join(path,'splits'))

			# Utilize multiple cores to analyze the images. Creates a pool of workers that has 1 fewer
			# worker than the number of cores available (so that the computer is still somewhat usable while this runs). 
			pool = multiprocessing.Pool(processes=num_cores)
			results = pool.map(process_split_helper, itertools.izip(splits, itertools.repeat(tds), itertools.repeat(root_dir)))
			pool.close()
			pool.join()


			####FOR DEVELOPMENT: 
			# Add a method (started below) to read all of the classified images in the case that 
			# every split has already been processed and deleted.
			# results = []
			# # If the classified image already exists, don't reclassify. Open the existing classification, 
			# # so that we can collect and return the pixel count data from that image.
			# output_dir = os.path.join(path,'splits','processed')
			# completed_files = os.listdir(output_dir)
			# for completed_name in completed_files:
			# 	if 'classified.h5' in completed_name:
			# 		try:
			# 			print "Reading classified: %s" %completed_name[:-18]
			# 			classified_file = h5py.File(os.path.join(output_dir,completed_name),'r')
			# 			classified_image = classified_file['classified'][:]
			# 			classified_file.close()
			# 			pixel_counts = utils.count_features(classified_image)
			# 			pixel_counts = list(pixel_counts)
			# 			results.append((completed_name[:-18], pixel_counts))
			# 		except IOError:
			# 			print "Corrupted file: %s" %completed_name
			######

			# If something was returned by the processing function, compile those results
			if results:

				#### Compile the output of all image splits into one csv entry

				# Group stats from each image into a dictionary {image: stats}
				compiled_stats = {}
				while len(results) > 0:
					# Grab one entry from the results list
					im_name,pixel_counts = results.pop()
					# Remove the split number
					im_name = im_name[:-8]

					# If the image is already in the dictionary, add to existing
					if im_name in compiled_stats:
						for i in range(len(compiled_stats[im_name])):
							compiled_stats[im_name][i] += pixel_counts[i]
					# Otherwise create a new entry
					else:
						compiled_stats[im_name] = pixel_counts

				# Write all of the dictionary keys (images) to the csv file
				for key in compiled_stats:
					utils.save_results("classification_results", root_dir, key, compiled_stats[key])
				
				#### Compile the split images back into a single image
				print "Recompiling image splits..."
				clsf_path = os.path.join(path,"splits","processed")	#clsf == classified
				clsf_splits_all = list(get_image_paths(clsf_path,keyword='classified.h5'))

				results_path = os.path.join(path,"classification_results")
				if not os.path.isdir(results_path):
					os.makedirs(results_path)

				while len(clsf_splits_all) > 0:

					# Pull out a single image at a time
					single_image = [clsf_splits_all.pop()]		#Initialize with the next split
					# Loop through remaining images to find matches ([:-22] trims extensions)
					num_splits = int(single_image[0][-16:-14])
					i = 0
					while len(single_image) < num_splits:
						if clsf_splits_all[i][:-22] in single_image[0]:
 							single_image.append(clsf_splits_all.pop(i))
							i-=1

						# Prevent infinite loops in case theres something wrong
						i+=1
						if i > len(clsf_splits_all + single_image):
							break

					print "Compiling: %s" %single_image[0][:-22]
					utils.stitch(single_image, save_path=results_path)

				print "Done."


# Code from http://chriskiehl.com/article/parallelism-in-one-line/
# Returns a list of .h5 files in the given folder.
def get_image_paths(folder,keyword='.h5'):
	return (os.path.join(folder, f)
		for f in os.listdir(folder)
		if keyword in f)

# multiprocessing pool.map will only accept one arguement, therefore we 
# need a helper funciton to pass all of the required arguements to 
# process_split
def process_split_helper(args):
	return process_split(*args)

def process_split(filename,training_set,save_path):

	# filename = split_data[0]
	tds = training_set

	# base is the directory containing the raw splits
	base, fname = os.path.split(filename)

	# output_dir is the location where the segmented and classified images are saved.
	output_dir = os.path.join(base,'processed')

	# Pause for a brief time before starting the first process. This prevents overlapping text output
	# by staggering the start time of these processes. 
	time.sleep(round(random.random(),1))

	# If the classified image already exists, don't reclassify. Open the existing classification, 
	# so that we can collect and return the pixel count data from that image.
	if os.path.isdir(output_dir):
		completed_files = os.listdir(output_dir)
		for completed_name in completed_files:
			if os.path.splitext(fname)[0] in completed_name and 'classified.h5' in completed_name:
				print "Already classified: %s" %fname
				try:
					classified_file = h5py.File(os.path.join(output_dir,completed_name),'r')
					classified_image = classified_file['classified'][:]
					classified_file.close()
					pixel_counts = utils.count_features(classified_image)
					pixel_counts = list(pixel_counts)
					return os.path.splitext(fname)[0], pixel_counts
				except IOError:
					print "Corrupted file (reclassifying): %s" %completed_name



	seg_time = time.clock()

	# ----------------------------
	# Optional section to save a color image for only a subset of all image 
	# splits - chosen randomly. 
	# c_check_int = int(round(random.random()*5,0))
	# if c_check_int == 1:
	# 	c_check = True
	# else:
	# 	c_check = False
	# ----------------------------


	# Segment the image
	print "Segmenting image: %s" %fname
	segment_image(base, fname, color_check=True)#c_check)
	print "Segment finished: %s: %f" %(fname, time.clock() - seg_time)

	segment = os.path.splitext(fname)[0] + '_segmented.h5'
	if os.path.isfile(os.path.join(output_dir, segment)):

		# Classify the split image
		class_time = time.clock()
		print "Classifying image: %s" %segment
		image_name, pixel_counts = classify_image(os.path.join(output_dir, segment), tds)
		print "Classification finished: %s: %f" %(segment,time.clock()-class_time)

		# Save the results to a common file
		utils.save_results("classification_results_raw", save_path, image_name, pixel_counts)

		# #Remove the split used in segmentation
		# print "Cleaning up split: Removed " + fname
		# os.remove(os.path.join(base,fname))

		# Remove the segmented image
		print "Cleaning up segments: Removed " + segment
		os.remove(os.path.join(output_dir,segment))
		return image_name, pixel_counts
	else:
		print "Skipped classification of: %s" %segment
		# Remove the split used in segmentation
		# print "Cleaning up split: Removed " + fname
		# os.remove(os.path.join(base,fname))


if __name__ == "__main__":
	main()
