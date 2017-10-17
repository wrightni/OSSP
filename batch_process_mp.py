#Batch Process
#Usage: Fully processes all images in the given directory with the given training data. 

#### Lite version of batch_process_mp.py
# This version will only output a single file in the given output directory
# Does not compile results into a csv, or produce a visual image of the results

# Changes from master:
# added output dir argument
# removed creation of output csv
# removed deletion of hidden files and folders (this needs to be changed 
#	in the main version as well)
# added a verbose flag to suppress/allow console output
# removes more of the intermediate files along the way and cleans up 
#	all temp folders at the end
# will always reclassify any images that it is given. does not check for ones
# 	that have already been classified. 
# 

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
	parser.add_argument("-s", "--splits", metavar='int', type=int, default=4,
						help="number of subdividing splits to preform on raw image")
	parser.add_argument("-p", "--parallel", metavar='int', type=int, default=1,
						help='''number of parallel processes to run. 
						It is typically prudent to leave at least 1 core unused. 
						Beware of memory usage requirements for each process.''')

	parser.add_argument("-o", "--output_dir", type=str, default=None,
						help="directory to place output results.")
	parser.add_argument("-v", "--verbose", action="store_true",
						help="display text information and progress")


	#### Parse Arguments
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

	if args.output_dir is None:
		output_dir = os.path.join(root_dir, 'classified')
	else:
		output_dir = args.output_dir
	verbose = args.verbose

	# Make sure the user doesn't try to use more cores than they have. 
	if num_cores > multiprocessing.cpu_count():
		num_cores = multiprocessing.cpu_count()-1

	#### Process images in the given directory.

	# List of directories in the root
	dates = next(os.walk(root_dir))[1]

	for path_, directories, files in os.walk(root_dir):

		# I think I need to remove this. Or change it to just skip hidden folders
		#  so that I dont accidentally delete things on other peoples systems
		# utils.remove_hidden(directories)
		# utils.remove_hidden(files)

		#### Split Each .tif

		# Loop through contents in the date directory
		for file in files:

			# Find .tif or .jpg formatted files
			if os.path.splitext(file)[1].lower() == '.tif' or os.path.splitext(file)[1].lower() == '.jpg':
				# We know we're working with an image now
				image_name = file
				# split_flag, true if we are going to split this image, false if it has already been split
				split_flag = True
				
				split_dir = os.path.join(path_,'splits')
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
					if verbose: 
						print "Splitting image: %s" %os.path.join(path_,image_name)
						print "-"*80
					split_image(path_, image_name, image_type, number_of_splits=num_splits)
				else:
					if verbose: print "Already Split: %s" %os.path.join(path_,image_name)

		# If there is a directory with splits, continue to process
		if os.path.isdir(os.path.join(path_,'splits')):

			#### Load Training Data
			tds = utils.load_tds(tds_file,tds_list)
		
			#### Analyze Splits

			# Find the list of splits that we need to process
			splits = get_image_paths(os.path.join(path_,'splits'))

			# Utilize multiple cores to analyze the images. Creates a pool of workers that has 1 fewer
			# worker than the number of cores available (so that the computer is still somewhat usable while this runs). 
			pool = multiprocessing.Pool(processes=num_cores)
			results = pool.map(process_split_helper, itertools.izip(splits, itertools.repeat(tds), 
				itertools.repeat(root_dir), itertools.repeat(verbose)))
			pool.close()
			pool.join()

			# If something was returned by the processing function, compile those results
			if results:

				#### Compile the split images back into a single image
				if verbose: print "Recompiling image splits..."
				clsf_path = os.path.join(path_,'splits','processed')	#clsf == classified
				clsf_splits_all = list(get_image_paths(clsf_path,keyword='classified.h5'))

				# results_path = os.path.join(path,"classification_results")

				if not os.path.isdir(output_dir):
					os.makedirs(output_dir)

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

						# Prevent infinite loops in case something is wrong
						i+=1
						if i > len(clsf_splits_all + single_image):
							break

					if verbose: print "Compiling: %s" %single_image[0][:-22]
					utils.stitch(single_image, save_path=output_dir)

				# Delete the temp file folder
				shutil.rmtree(os.path.join(path_,'splits'))
				if verbose: print "Done."


# Code from http://chriskiehl.com/article/parallelism-in-one-line/
# Returns a list of .h5 files in the given folder.
def get_image_paths(folder,keyword='.h5'):
	return (os.path.join(folder, f)
		for f in os.listdir(folder)
		if keyword in f)

# multiprocessing pool.map will only accept one argument, therefore we 
# need a helper function to pass all of the required arguments to 
# process_split
def process_split_helper(args):
	return process_split(*args)

def process_split(filename,training_set,save_path, verbose):

	# filename = split_data[0]
	tds = training_set

	# base is the directory containing the raw splits
	base, fname = os.path.split(filename)

	# output_dir is the location where the segmented and classified images are saved.
	processed_dir = os.path.join(base,'processed')

	seg_time = time.clock()

	# Segment the image
	if verbose: print "Segmenting image: %s" %fname
	segment_image(base, fname)
	if verbose: print "Segment finished: %s: %f" %(fname, time.clock() - seg_time)

	segment = os.path.splitext(fname)[0] + '_segmented.h5'
	if os.path.isfile(os.path.join(processed_dir, segment)):

		# Classify the split image
		class_time = time.clock()
		if verbose: print "Classifying image: %s" %segment
		image_name, pixel_counts = classify_image(os.path.join(processed_dir, segment), tds)
		if verbose: print "Classification finished: %s: %f" %(segment,time.clock()-class_time)

		# Maybe don't need to clean up temp files here because we are 
		#	removing the entire temp directory later

		# #Remove the split used in segmentation
		os.remove(os.path.join(base,fname))

		# Remove the segmented image
		os.remove(os.path.join(processed_dir,segment))

		return image_name, pixel_counts
	else:
		if verbose: print "Skipped classification of: %s" %segment
		# Remove the split used in segmentation
		# print "Cleaning up split: Removed " + fname
		# os.remove(os.path.join(base,fname))


if __name__ == "__main__":
	main()
