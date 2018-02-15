#Batch Process
#Usage: Fully processes all images in the given directory with the given training data. 

#### Lite version of batch_process_mp.py
# This version will only output a single file in the given output directory
# Does not compile results into a csv, or produce a visual image of the results

# Changes from master:
# added output dir argument
# removed creation of output csv
# removed deletion of hidden files and folders (this needs to be changed 
#   in the main version as well)
# added a verbose flag to suppress/allow console output
# removes more of the intermediate files along the way and cleans up 
#   all temp folders at the end
# will always reclassify any images that it is given. does not check for ones
#   that have already been classified. 
# 

import os
import shutil
import argparse
import multiprocessing
import time
import h5py
import numpy as np

from preprocess import prepare_image
from segment import segment_image
from classify import classify_image

from lib import utils

import matplotlib.pyplot as plt
from skimage import filters, morphology, feature, exposure, segmentation, future


def main():

    #### Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", 
                        help=('''directory path containing date directories of 
                        images to be processed'''))
    parser.add_argument("image_type", type=str, choices=['srgb','wv02_ms','pan'],
                        help="image type: 'srgb', 'wv02_ms', 'pan'")
    parser.add_argument("training_dataset",
                        help="training data file")
    parser.add_argument("--training_label", type=str, default=None,
                        help="name of training classification list")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="directory to place output results.")
    parser.add_argument("-s", "--splits", metavar='int', type=int, default=1,
                        help="number of subdividing splits to preform on raw image")
    parser.add_argument("-p", "--parallel", metavar='int', type=int, default=1,
                        help='''number of parallel processes to run. 
                        It is typically prudent to leave at least 1 core unused. 
                        Beware of memory usage requirements for each process.''')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display text information and progress")

    #### Parse Arguments
    args = parser.parse_args()

    # System filepath that contains the directories or files for batch processing
    user_input = args.input_dir
    if os.path.isdir(user_input):
        src_dir = user_input
        src_file = ''
    elif os.path.isfile(user_input):
        src_dir,src_file = os.path.split(user_input)
    else:
        raise IOError('Invalid input')
    # Image type, choices are 'srgb', 'pan', or 'wv02_ms'
    image_type = args.image_type
    # File with the training data
    tds_file = args.training_dataset
    # Default tds label is the image type
    if args.training_label is None:
        tds_label = image_type
    else:
        tds_label = args.training_label
    # Default output directory (if not provided)
    if args.output_dir is None:
        dst_dir = os.path.join(src_dir, 'classified')
    else:
        dst_dir = args.output_dir
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    num_splits = args.splits
    num_threads = args.parallel
    verbose = args.verbose

    # Make sure the user doesn't try to use more cores than they have. 
    if num_threads > multiprocessing.cpu_count():
        num_threads = multiprocessing.cpu_count()-1

    # Directory where temporary files are saved
    working_dir = os.path.join(src_dir, 'splits')

    ### Prepare a list of images to be processed based on the user input
    # List of task objects based on the files in the input directory
    # Each task is an image to process, and has a subtask for each split
    #   of that image. 
    task_list = utils.create_task_list(os.path.join(src_dir,src_file), 
                                        dst_dir, num_splits)
    ### Load Training Data
    tds = utils.load_tds(tds_file,tds_label)

    for task in task_list:

        # Skip this task if it is already marked as complete
        if task.is_complete():
            continue

        # If the image has not yet been split or if no splitting was requested,
        # proceed to the preprocessing step.
        if not task.is_split() or num_splits == 1:
            image_name = task.get_id()
            image_data, meta_data = prepare_image(src_dir, image_name, image_type, 
                                        output_path=working_dir,
                                        number_of_splits=num_splits, 
                                        verbose=verbose)
            block_dims = meta_data[0]
            image_date = meta_data[1]

        pixel_counts = [0,0,0,0,0]
        classified_image = []
        # Loop until all subtasks are complete.
        # Breaks when task.get_next_subtask() returns None (all subtasks complete)
        #   or if the task is complete.
        while True:

            if task.is_complete():
                break
            elif task.has_subtask():
                subtask = task.get_next_subtask()

                if subtask == None:
                    break
                # If there is a subtask, the image data is stored in a split on the
                #   drive. Subtask == {} when there are no subtasks.
                image_data = os.path.join(working_dir,subtask) + '.h5'
                with h5py.File(image_data,'r') as f:
                    block_dims = f.attrs.get("Block Dimensions")
                    image_date = f.attrs.get("Image Date")
            else:
                subtask = task.get_id()

            ## Segment image
            seg_time = time.clock()
            if verbose: print("Segmenting image: %s" %subtask)
            image_data, segmented_blocks = segment_image(image_data, 
                                image_type=image_type,
                                threads=num_threads, 
                                verbose=verbose)
            if verbose: print("Segment finished: %s: %f" 
                              %(subtask, time.clock() - seg_time))

            ###
            # from lib import debug_tools
            # debug_tools.display_watershed(image_data, segmented_blocks)
            # quit()
            ####

            ## Classify image
            class_time = time.clock()
            if verbose: print("Classifying image: %s" %subtask)
            classified_blocks = classify_image(image_data, segmented_blocks, tds, 
                                 [image_type,image_date], threads=num_threads, 
                                 verbose=verbose)
            if verbose: print("Classification finished: %s: %f" 
                              %(subtask,time.clock()-class_time))

            ## Hold onto the output of this subtask
            clsf_split = utils.compile_subimages(classified_blocks,block_dims[0],
                                             block_dims[1])
            
            # Save the results to the temp folder if there is more than 1 split
            if num_splits > 1:
                with h5py.File(os.path.join(working_dir,subtask)+'_classified.h5',
                               'w') as f:
                    f.create_dataset('classified',data=clsf_split,
                                     compression='gzip',compression_opts=3)
            else:
                classified_image = clsf_split
            
            # Add the pixel counts from this classified split to the 
            #   running total.
            pixel_counts_split = utils.count_features(clsf_split)
            for i in range(len(pixel_counts)):
                pixel_counts[i] += pixel_counts_split[i]

            # Mark this subtask as complete. This sets task.complete to True
            #   if there are no subtasks. 
            task.update_subtask(subtask)


        ####
        # from lib import debug_tools
        # while True:
        #     blk = raw_input("Enter block number: ")
        #     if blk == 'n':
        #         break
        #     else:
        #         blk = int(blk)
        #     im = image_data[1][blk]
        #     seg_im = segmented_blocks[blk]
        #     clsf_im = classified_blocks[blk]
        #     debug_tools.display_image(im,seg_im,clsf_im,1)


        ####
        # Write the total pixel counts to the database (or csv)
        utils.write_to_csv(os.path.join(dst_dir,task.get_id()), dst_dir, subtask, 
                         pixel_counts)

        ## Writing the results to a sqlite database. (Only works for 
        #   a specific database structure that has already been created)
        # db_name = 'ImageDatabase.db'
        # db_dir = '/media/sequoia/DigitalGlobe/'
        # image_name = task.get_id()
        # image_name = os.path.splitext(image_name)[0]
        # image_id = image_name.split('_')[2]
        # part = image_name.split('_')[5]
        # utils.write_to_database(db_name, db_dir, image_id, part, pixel_counts)

        #### Compile the split images back into a single image
        if verbose: print("Recompiling: %s" %task.get_id())
        image_name = os.path.splitext(image_name)[0]

        # Create a sorted list of the tasks. Then create the correct filename
        #   for each split saved on the drive.
        #  to each id
        if num_splits > 1:
            clsf_splits = []
            task_list = task.get_tasklist()
            task_list.sort()
            for task_id in task_list:
                cname = os.path.join(working_dir, task_id) + "_classified.h5"
                clsf_splits.append(cname)
            utils.stitch(clsf_splits, save_path=dst_dir)
        else:
            with h5py.File(os.path.join(dst_dir,image_name)+'_classified.h5', 'w') as f:
                f.create_dataset('classified',data=clsf_split,
                                 compression='gzip',compression_opts=9)
            # Save color image for viewing
            utils.save_color(classified_image, 
                             os.path.join(dst_dir,image_name)+'.png')

        ### Remove temp folders?
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)

        if verbose: print("Done")


if __name__ == "__main__":
    main()
