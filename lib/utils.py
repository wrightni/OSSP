import h5py
import os
import csv
import math
import itertools
import sqlite3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mimg
from ctypes import *

#
valid_extensions = ['.tif','.tiff','.jpg']

class Task:

    def __init__(self,name):
        self.task_id = name
        self.subtasks = {}
        self.split = False      # True if image has been split
        self.complete = False
    
    def get_id(self):
        return self.task_id

    def change_id(self, name):
        self.task_id = name

    def add_subtask(self, subtask_id):
        self.subtasks[subtask_id] = False

    def update_subtask(self, subtask_id):
        if self.has_subtask():
            self.subtasks[subtask_id] = True
        else:
            self.mark_complete()

    def mark_split(self):
        self.split = True

    def mark_complete(self):
        self.complete = True

    def is_split(self):
        return self.split

    def is_complete(self):
        return self.complete

    def get_tasklist(self):
        return self.subtasks.keys()

    def get_unfinished(self):
        # Returns a list of unfinished subtasks
        return [st for st in self.subtasks.keys() if not self.subtasks[st]]

    def has_subtask(self):
        if self.subtasks == {}:
            return False
        else:
            return True

    def get_subtask_status(self, subtask_id):
        return self.subtasks[subtask_id]

    # Returns the id and a command for the next subtask
    #  that needs to be completed
    def get_next_subtask(self):

        if self.complete:
            return None
        # If there are no subtasks (ie num_splits == 1), return
        #   the name of the task
        elif self.subtasks == {}:
            return None
        # Otherwise return the next unfinished subtask
        else:
            # st is the split id
            for st in self.subtasks.keys():
                if self.subtasks[st] == False:
                    return st

        # If we get to this point, all splits are classified,
        #  but need to be compiled (otherwise complete would be True)
        # return self.task_id, 'compile'
        return None


def create_task_list(src_dir, dst_dir, num_splits):

    task_list = []
    # If the input is a file, return that file as the only task
    if os.path.isfile(src_dir):
        src_dir,src_file = os.path.split(src_dir)
        task = Task(src_file)
        if num_splits > 1:
            for i in range(1,num_splits+1):
                sname = os.path.splitext(src_file)[0] \
                    + "_s{0:02d}of{1:02d}".format(i,num_splits)
                task.add_subtask(sname)
        return [task]

    for path_, directories, files in os.walk(src_dir):

        # Skip hidden directories
        if path_[0] == '.':
            continue

        # Loop through contents in the current directory
        for file in files:

            # Skip hidden files
            if file[0] == '.':
                continue

            image_name,ext = os.path.splitext(file)
            # Check that the file is .tif or .jpg format
            ext = ext.lower()
            if ext not in valid_extensions:
                continue
            
            ## Create the task object for this image
            task = Task(file)
            # Add a subtask for each image split
            if num_splits > 1:
                for i in range(1,num_splits+1):
                    sname = image_name \
                        + "_s{0:02d}of{1:02d}".format(i,num_splits)
                    task.add_subtask(sname)

            ## Check the output directory for completed files
            if os.path.isdir(dst_dir):
                clsf_imgs = os.listdir(dst_dir)
                # Finished images have a consistant naming structure:
                target_name = image_name + '_classified.h5'
                for img in clsf_imgs:
                    # Set this task to complete if we find the finished image
                    if img == target_name:
                        task.mark_complete()
                
                ## Skip to the next image if this task is complete
                if task.is_complete():
                    continue

                # If the image is not complete, look for finished subtasks
                if num_splits > 1:
                    # Look for the split name header:
                    target_name = image_name + '_s'
                    clsf_splits = [cs for cs in clsf_imgs if target_name in cs]
                    for cs in clsf_splits:
                        sname = cs.rsplit('_',1)[0] #remove trailing modifier
                        task.update_subtask(sname)

            ## Continue to determine task status

            # Check if the image has already been split by looking in the
            #   'splits' subdirectory
            # If theres no split directory, we assume the image has not
            #   been split
            split_dir = os.path.join(path_,'splits')

            if os.path.isdir(split_dir):
                
                # list the files in the splits directory
                split_dir_files = os.listdir(split_dir)
                # Loop through all files; find any that match 
                #   the current image name. Potentially revisit this later
                #   as this might trigger on files other than splits. 
                
                # split names have a consistant structure
                sname_header = image_name + '_s'
                # List the splits of the current image
                splits = [os.path.splitext(s)[0] for s in split_dir_files if sname_header in s]
                # List of the unprocessed splits
                unfinished = task.get_unfinished()
                # Compare the splits that exist with the ones we need, if there
                #  are any that need to be processed, but do not exist, the
                #  whole image needs to be split.
                marker = False
                for sname in unfinished:
                    if sname not in splits:
                        marker = True
                # If all the needed splits exist, then mark the split task True.
                if marker == False:
                    task.mark_split()

            task_list.append(task)

    return task_list


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
    # Remove feature lists that don't have an associated label
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
    x = 0
    while 5 in label_vector:
        i = label_vector.index(5)
        label_vector.pop(i)
        training_feature_matrix.pop(i)
        x+=1

    # Combine the label vector and training feature matrix into one variable. 
    tds = [label_vector,training_feature_matrix]

    return tds

#### Save classification results
def write_to_csv(csv_name, path, image_name, pixel_counts):
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

    num_pixels = 1  #Prevent division by 0
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

def write_to_database(db_name, path, image_id, part, pixel_counts):
    '''
    INPUT:
        db_name: filename of the database
        path: location where the database is stored
        pixel_clounts: number of pixels in each classification category
            [snow, gray, melt, water, shadow]

    Writes the classification pixel counts to the database at the image_id entry
    NOTES:
        For now, this overwrites existing data.
    FUTURE:
        Develop method that checks for existing data and appends the current
            data to that, record which parts contributed to the total.

    '''
    # Convert pixel_counts into percentages and total area
    area = 1  #Prevent division by 0
    for i in range(len(pixel_counts)):
        area +=  pixel_counts[i]
    percentages = []
    for i in range(len(pixel_counts)):
        percentages.append(float(pixel_counts[i]/area))
    # Open the database
    conn = sqlite3.connect(os.path.join(path,db_name))

    # Update the entry at image_id with the given pixel counts
    conn.execute("UPDATE DigitalGlobe \
                 SET AREA = {0:d}, SNOW = {1:f}, GRAY = {2:f}, MP = {3:f}, \
                     OW = {4:f}, PART = {5:s} \
                 WHERE NAME = '{6:s}' \
                ".format(int(area), percentages[0], percentages[1], 
                         percentages[2], percentages[3], part, image_id)
                )
    # Commit the changes
    conn.commit()
    # Close the database
    conn.close()

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
    #   This method relies on floating point precision, but
    #   will be accurate within the scope of this method
    root = math.sqrt(len(image_files))
    if int(root) != root:
        print "Incomplete set of images!"
        return None

    classified_list = []

    image_files.sort()

    ## Read the classified data and the original image data
    for image in image_files:
        with h5py.File(image,'r') as inputfile:
            classified_image = inputfile['classified'][:]
            classified_list.append(classified_image)


    # Find the right dimensions for stitching the images back together
    box_side = int(math.sqrt(len(classified_list)))

    # Stitch the classified image back together
    full_classification = compile_subimages(classified_list,box_side,box_side)

    # if os.path.isdir(save_path):
    #     output_name = os.path.join(save_path, os.path.split(image_files[0])[1][:-18])
    #     fout = h5py.File(output_name + "_classified.h5",'w')
    #     fout.create_dataset('classified',data=full_classification,compression='gzip',compression_opts=9)
    #     fout.close()
    # else:
    #     save_color(full_classification, image_files[0][:-18] + "_classified_image.png")
    #     fout = h5py.File(image_files[0][:-18] + "_classified.h5",'w')
    #     fout.create_dataset('classified',data=full_classification,compression='gzip',compression_opts=9)
    #     fout.close()

    return full_classification


def compile_subimages(subimage_list, num_x_subimages, num_y_subimages, bands=1):
    '''
    Compiles the subimages (i.e. blocks) of a split into one raster
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
        compiled_image = np.zeros([num_y_subimages*y_size,
                                   num_x_subimages*x_size,
                                   bands],dtype='uint8')

        counter = 0
        for y in range(num_y_subimages):
            for x in range(num_x_subimages):
                compiled_image[y*y_size:(y+1)*y_size,
                               x*x_size:(x+1)*x_size,
                               :] = subimage_list[counter]
                counter += 1
    else:
        compiled_image = np.zeros([num_y_subimages*y_size,
                                   num_x_subimages*x_size],
                                   dtype='uint8')
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
        empty_color = [.1,.1,.1]        #Almost black
        snow_color = [.9,.9,.9]         #Almost white
        pond_color = [.31,.431,.647]    #Blue
        gray_color = [.5,.5,.5]         #Gray
        water_color = [0.,0.,0.]        #Black
        shadow_color = [1.0, .545, .0]  #Orange
        cloud_color = [.27, .15, .50]       #Purple

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


def get_image_paths(folder,keyword='.h5',strict=True):
    '''
    Code from http://chriskiehl.com/article/parallelism-in-one-line/
    Returns a list of .h5 files in the given folder.
    Strict flag restricts keyword to the extension, non strict will find the
        keyword anywhere in the filename
    '''
    if strict:
        return (os.path.join(folder, f)
            for f in os.listdir(folder)
            if (keyword in os.path.splitext(f)[1].lower()
                and os.path.splitext(f)[0][0] != '.'))
    else:
        return (os.path.join(folder, f)
                for f in os.listdir(folder)
                if (keyword in f.lower()
                    and os.path.splitext(f)[0][0] != '.'))

# Remove hidden folders and files from the given list of strings (mac)
def remove_hidden(folder):
    i = 0
    while i < len(folder):
        if folder[i][0] == '.':
            folder.pop(i)
        else:
            i+=1
    return folder

# Combines multiple bands (RBG) into one 3D array
# Adapted from:  http://gis.stackexchange.com/questions/120951/merging-multiple-16-bit-image-bands-to-create-a-true-color-tiff
# Useful band combinations: http://c-agg.org/cm_vault/files/docs/WorldView_band_combs__2_.pdf
def create_composite(band_list, dtype=np.uint8):
    img_dim = np.shape(band_list[0])
    num_bands = len(band_list)
    img = np.zeros((img_dim[0], img_dim[1], num_bands), dtype=dtype)
    for i in range(num_bands):
        img[:,:,i] = band_list[i]
    
    return img

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