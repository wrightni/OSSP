#title: Training Set Creation for Random Forest Classification
#author: Nick Wright
#Inspired by: Justin Chen

#purpose: Creates a GUI for a user to identify watershed superpixels of an image as
#        melt ponds, sea ice, or open water to use as a training data set for a 
#        Random Forest Classification method. 


import Tkinter as tk
import tkFont
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from ctypes import *
import gdal
from sklearn.ensemble import RandomForestClassifier
from select import select
import sys

import preprocess as pp
from segment import segment_image
from lib import utils
from lib import attribute_calculations as attr_calc


class print_color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


#This class structure logic might be a little far-fetched
class TrainingWindow:

    def __init__(self, original_image, secondary_image, segment_list, image_name, label_vector, feature_matrix, 
        im_type, im_metadata, savepath):
        """Constructor"""
        #Variables
        self.parent = tk.Tk()
        self.original_image = original_image
        # In mode 1, secondary image is the watershed segments
        # In mode 2, secondary image is the classified image
        self.secondary_image = secondary_image
        self.label_vector = label_vector
        self.feature_matrix = feature_matrix
        # remove "_segmented" from name
        image_name_split = str(image_name).rsplit('_',1)
        if "segmented" in image_name_split[1]:
            self.image_name = image_name_split[0]
        else:
            self.image_name = os.path.splitext(image_name)[0]
        # Format of the ID values in segment_list
        # 1: ["image_name", segment_number]
        # 2: [x,y]
        self.segment_list = segment_list
        self.im_type = im_type                  # Input image type
        self.im_date = im_metadata[0]
        self.wb_ref = im_metadata[1:9]
        self.br_ref = im_metadata[9:]
        self.savepath = savepath
        self.tracker = 0                        # Number of segment sets added from the current image
        # **********
        self.rfc = RandomForestClassifier(n_estimators=100)
        self.rfc.fit(self.feature_matrix[:len(self.label_vector)], self.label_vector)
        self.zoom_win_x = 0
        self.zoom_win_y = 0
        # **********
        # If there is no existing segment list or the segment list and label list are the same length
        #  initialize the subimage index and pixel buffer with random integers. 
        if segment_list == [] or len(label_vector) == len(segment_list):
            # Find some segments to add to the 'to-do' list
            self.add_segments()
            # Fill sp_buffer with the first segment's info
            self.sp_buffer = segment_list[len(label_vector)][1]
        # Pull the first unclassified segment from segment list, then parse the needed number from that id
        # based on the mode we are using.
        self.sp_buffer = segment_list[len(label_vector)][1]   # current super pixel
        self.parent.title("Training Creation")
        frame = tk.Frame(self.parent)
        frame.pack(side='left')
        buttons = tk.Frame(self.parent)
        buttons.pack(side='right')
        
        #Defining the buttons
        prevWsBtn = tk.Button(self.parent, text="Previous Segment",width=16, height=2, command=lambda: self.previous_super_pixel())
        prevWsBtn.pack(in_=buttons, side='top')
        spacer = tk.Button(self.parent, text="  ", width=16, height=2)
        spacer.pack(in_=buttons,side="top")
        waterBtn = tk.Button(self.parent, text="Open Water", width=16, height=2, highlightbackground='#000000',
            command=lambda: self.classify("water"))
        waterBtn.pack(in_=buttons, side='top')
        meltBtn = tk.Button(self.parent, text="Melt Pond", width=16, height=2, highlightbackground='#4C678C',
            command=lambda: self.classify("melt"))
        meltBtn.pack(in_=buttons, side='top')
        grayIceBtn = tk.Button(self.parent, text="Dark and Thin Ice", width=16, height=2, highlightbackground='#D2D3D5',
            command=lambda: self.classify("gray"))
        grayIceBtn.pack(in_=buttons, side='top')
        # iceBtn = tk.Button(self.parent, text="Thick Ice", width=16, height=2, command=lambda: self.classify("ice"))
        # iceBtn.pack(in_=buttons, side='top')
        snowBtn = tk.Button(self.parent, text="Snow/Ice", width=16, height=2, 
            command=lambda: self.classify("snow"))
        snowBtn.pack(in_=buttons, side='top')
        shadowBtn = tk.Button(self.parent, text="Dark Pond", width=16, height=2, highlightbackground='#FF9200',
            command=lambda: self.classify("shadow"))
        shadowBtn.pack(in_=buttons, side='top')
        unknownBtn = tk.Button(self.parent, text="Unknown / Mixed", width=16, height=2, command=lambda: self.classify("unknown"))
        unknownBtn.pack(in_=buttons, side='top')
        ##
        autoBtn = tk.Button(self.parent, text="Autorun", width=16, height=2, command=lambda: self.autorun())
        autoBtn.pack(in_=buttons, side='top')
        ##
        spacer2 = tk.Button(self.parent, width=16, height=2)
        spacer2.pack(in_=buttons,side="top")
        nextBtn = tk.Button(self.parent, text="Next Image", width=16, height=2, command=lambda: self.exit_image())
        nextBtn.pack(in_=buttons, side='top')
        quitBtn = tk.Button(self.parent, text="Save and Quit", width=16, height=2, command=lambda: self.quit())
        quitBtn.pack(in_=buttons, side='top')

        self.parent.bind('1', lambda e: self.classify("snow")) 
        self.parent.bind('2', lambda e: self.classify("gray")) 
        self.parent.bind('3', lambda e: self.classify("melt")) 
        self.parent.bind('4', lambda e: self.classify("water")) 
        self.parent.bind('5', lambda e: self.classify("shadow"))
        self.parent.bind('<Tab>', lambda e: self.classify("unknown"))
        self.parent.bind('<BackSpace>', lambda e: self.previous_super_pixel())

        # self.parent.bind('<Button-2>', self.onclick)

        #Creating the canvas where the images will be
        self.fig = plt.figure(figsize=[10,10])
        self.fig.subplots_adjust(left=0.01,right=0.99,bottom=0.05,top=0.99,wspace=0.01,hspace=0.01)
        canvas = FigureCanvasTkAgg(self.fig, frame)
        # toolbar = NavigationToolbar2TkAgg(canvas, frame)
        canvas.get_tk_widget().pack(in_=frame, side='top')
        # toolbar.pack(in_=frame, side='top')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.next_super_pixel()
        self.parent.mainloop()
    
    #Displays the 3 images
    def displayImages(self):
    
        plt.clf()

        if self.im_type == 'wv02_ms':
            display_image = utils.create_composite([self.original_image[4, :, :],
                                                    self.original_image[2, :, :],
                                                    self.original_image[1, :, :]],
                                                   dtype=np.uint8)

        elif self.im_type == 'pan':
            display_image = utils.create_composite([self.original_image,
                                                    self.original_image,
                                                    self.original_image],
                                                   dtype=np.uint8)

        elif self.im_type == 'srgb':
            display_image = utils.create_composite([self.original_image[0, :, :,],
                                                    self.original_image[1, :, :,],
                                                    self.original_image[2, :, :,]],
                                                   dtype=np.uint8)

        ws_subimage = self.secondary_image

        current_sp = ws_subimage==self.sp_buffer      #array of 0 or 1 where 1 = current superpixel
        spPosition = np.nonzero(current_sp)           #returns the array position of the superpixel

        zoom_size = 100

        xMin = np.amin(spPosition[0]) - zoom_size
        xMax = np.amax(spPosition[0]) + zoom_size
        yMin = np.amin(spPosition[1]) - zoom_size
        yMax = np.amax(spPosition[1]) + zoom_size

        # Store the zoom window corner coordinates for reference in onclick()
        # xMin and yMin are defined backwards
        self.zoom_win_x = yMin
        self.zoom_win_y = xMin

        if xMin < 0:
            xMin = 0
        if xMax >= np.shape(display_image)[0]:
            xMax = np.shape(display_image)[0] - 1
        if yMin < 0:
            yMin = 0
        if yMax >= np.shape(display_image)[1]:
            yMax = np.shape(display_image)[1] - 1

        #Image 1 (Full zoomed out image)
        full_image = display_image

        #Image 2 (Zoomed in image, no highlighted segment)
        cropped_image = np.copy(display_image)
        cropped_image = cropped_image[xMin:xMax,yMin:yMax] #Cropping

        #Image 3 (Zoomed in image, with segment highlight)
        color_image = np.copy(display_image)
        color_image[:, :, 0][current_sp] = 255
        color_image[:, :, 2][current_sp] = 0
        color_image = color_image[xMin:xMax, yMin:yMax]

        # Text string to display the user's progress
        progress_counter = "Number Classified: {} / {} ({})".format(len(self.label_vector), len(self.segment_list), self.tracker)
        
        #Text instructions
        instructions = '''
%s \n
Open Water: Surface areas that had zero ice cover 
as well as those covered by an unconsolidated frazil 
or grease ice. \n
Melt Pond: Surface areas with water covering ice. 
 Areas where meltwater is trapped in isolated patches
 atop ice, and the optically similar submerged ice 
 near the edge of a floe. \n
Dark Ice: 
Freezing season: Surfaces of thin ice that are 
not snow covered, including nilas and young ice. 
Melt season: ice covered by saturated slush, 
but not completely submerged in water \n
Snow/Ice: Optically thick ice, and ice with a snow cover. \n
Shadow: Surfaces that are covered by a dark shadow. 
 \n
''' %(progress_counter)

        #Plotting onto the GUI
        ax = self.fig.add_subplot(2,2,1)
        ax.imshow(color_image,interpolation='None',vmin=0,vmax=255)
        ax.tick_params(axis='both',          # changes apply to the x-axis
                       which='both',         # both major and minor ticks are affected
                       bottom='off',         # ticks along the bottom edge are off
                       top='off',            # ticks along the top edge are off
                       left='off',
                       right='off',
                       labelleft='off',
                       labelbottom='off')
        ax.set_label('ax1')

        ax = self.fig.add_subplot(2,2,2)
        ax.imshow(cropped_image,interpolation='None',vmin=0,vmax=255)
        ax.tick_params(axis='both',          # changes apply to the x-axis
                       which='both',         # both major and minor ticks are affected
                       bottom='off',         # ticks along the bottom edge are off
                       top='off',            # ticks along the top edge are off
                       left='off',
                       right='off',
                       labelleft='off',
                       labelbottom='off')
        ax.set_label('ax2')

        ax = self.fig.add_subplot(2,2,3)
        ax.imshow(full_image,interpolation='None',vmin=0,vmax=255)
        ax.axvspan(yMin,
                   yMax,
                   1.-float(xMax)/np.shape(full_image)[0],
                   1.-float(xMin)/np.shape(full_image)[0],
                   color='red',
                   alpha=0.3)
        ax.set_xlim([0,np.shape(full_image)[1]])
        ax.tick_params(axis='both',          # changes apply to the x-axis
                       which='both',         # both major and minor ticks are affected
                       bottom='off',         # ticks along the bottom edge are off
                       top='off',            # ticks along the top edge are off
                       left='off',
                       right='off',
                       labelleft='off',
                       labelbottom='off')
        ax.set_label('ax3')

        ax = self.fig.add_subplot(2,2,4,adjustable='datalim',frame_on=False)
        ax.text(0.5,0.5,instructions,horizontalalignment='center',verticalalignment='center')
        ax.axis('off')

        #Updating the plots
        self.fig.canvas.draw()

    
    def next_super_pixel(self):
        
        # Format of the ID values in segment_list (mode 1)
        # 01234567890123
        # mmddpsiiiinnnn
        # m: month | d: day | p: part | s: split | i: subimage | n: segment number
        
        # If all of the segments in the predefined list have been classified already, 
        # present the user with a random new segment.

        # Add a new set of segments to the list, unless this process has already happened
        # more than x number of times (tracker)
        if len(self.label_vector) == len(self.segment_list):
            # if self.tracker >= 9:
            #     self.exit_image()
            #     return
            self.add_segments()
            # self.tracker += 1

        # The current segment is the next one that doesn't have an associated label
        current_segment = self.segment_list[len(self.label_vector)]

        # Exit the training window if we have finished the preset list of segments for this
        # image
        if os.path.splitext(current_segment[0])[0] != self.image_name:
            print "Finished image %s. Loading next image." %self.image_name
            self.exit_image()
            return

        self.sp_buffer = int(current_segment[1])

        self.displayImages()

    def previous_super_pixel(self):
        # Make sure this function returns null if there is no previous sp to go back to
        if len(self.label_vector) == 0:
            return
        else:
            # Remove the old classification
            self.label_vector.pop()
            self.tracker -= 1
            # Update the super pixel index
            current_segment = self.segment_list[len(self.label_vector)]
            if current_segment[0] != self.image_name:
                self.exit_image()
                return
            self.sp_buffer = int(current_segment[1])

            # Redraw the display window with the updated indicies. 
            self.displayImages()

    # Add some new segments if we've run out of ones to classify
    def add_segments(self):
        segments_to_add = []

        a = 0
        # Select random x,y coordinates from the input image, and pick the segment where the random
        #   pixel lands. This makes the selected segments representative of the average surface
        #   distribution within the image. This still wont work if the image has a minority of any
        #   particular surface type.
        while len(segments_to_add)<10:
            a += 1
            z,x,y = np.shape(self.original_image)
            i = np.random.randint(x)
            j = np.random.randint(y)
            # Find the segment label at the random pixel
            segment_id = self.secondary_image[i][j]
            sp_size = np.sum(self.secondary_image == segment_id)
            if sp_size >= 20:
                # Check for a duplicate segment already in the tds
                new_segment = [self.image_name,
                               "{}".format(segment_id)]
                if new_segment not in self.segment_list and new_segment not in segments_to_add:
                    segments_to_add.append(new_segment)

        print("Attempts: {}".format(a))
        self.segment_list += segments_to_add

        # **********
        self.rfc = RandomForestClassifier(n_estimators=100)
        self.rfc.fit(self.feature_matrix[:len(self.label_vector)], self.label_vector)
        # **********

    def onclick(self, event):
        if event.inaxes is not None:
            axes_properties = event.inaxes.properties()
            segment_id = -1

            # # If the mouse click was in the overview image
            if axes_properties['label'] == 'ax3':
                x = int(event.xdata)
                y = int(event.ydata)
                segment_id = self.secondary_image[y, x]

            # # Either of the top zoomed windows
            if axes_properties['label'] == 'ax1' or axes_properties['label'] == 'ax2':
                win_x = int(event.xdata)
                win_y = int(event.ydata)
                x = self.zoom_win_x + win_x
                y = self.zoom_win_y + win_y
                segment_id = self.secondary_image[y, x]

            if segment_id >= 0:
                print("You clicked at ({}, {}) in {}".format(x, y, axes_properties['label']))
                print("Segment id: {}".format(segment_id))
                new_segment = [self.image_name,
                               "{}".format(segment_id)]
                if new_segment not in self.segment_list:
                    self.segment_list = self.segment_list[:len(self.label_vector)]
                    self.segment_list.append(new_segment)

                    self.next_super_pixel()
                else:
                    print("This segment has already been labeled")


    def autorun(self):
        segment_id = int(self.segment_list[len(self.label_vector):][0][1])

        feature_array = attr_calc.analyze_ms_image(self.original_image,
                                                   self.secondary_image,
                                                   self.wb_ref,
                                                   self.br_ref,
                                                   segment_id=segment_id)
        # attribute_calculations returns a 2d array, but we only want the 1d list of features.
        feature_array = feature_array[0]
        print "~" * 80
        pred, proba = self.print_prediction(feature_array)
        if 0.90 < proba < 0.96:
            timeout = 4
            print(print_color.BOLD + "Label if incorrect:" + print_color.END)
        elif proba < .9:
            timeout = 10
            print(print_color.BOLD + print_color.RED + "Label if incorrect:" + print_color.END)
        else:
            timeout = 0.5

        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            s = sys.stdin.readline()
            try:
                s = int(s)
            except ValueError:
                print("Ending autorun.")
                return
            if 0 <= s < 6:
                label = s
                print("Assigning label {} instead.".format(label))
            else:
                print("Ending autorun.")
                return
        else:
            label = pred
            print("No input. Assigning label: {}".format(label))

        self.label_vector.append(label)
        self.tracker += 1

        if len(self.feature_matrix) == len(self.label_vector)-1:
            #Adding all of the features found for this watershed to the main matrix
            self.feature_matrix.append(feature_array)
        else:
            print("Recalculated Feature.")
            self.feature_matrix[len(self.label_vector)-1] = feature_array

        self.next_super_pixel()
        self.parent.after(100, self.autorun)


    #Assigning the highlighted superpixel a classification
    def classify(self, keyPress):
        segment_id = int(self.segment_list[len(self.label_vector):][0][1])
        print(segment_id)
        # Note that we classified one more image
        if keyPress == "snow":
            self.label_vector.append(1)
        elif keyPress == "gray":
            self.label_vector.append(2)
        elif keyPress == "melt":
            self.label_vector.append(3)
        elif keyPress == "water":
            self.label_vector.append(4)
        elif keyPress == "shadow":
            self.label_vector.append(5)
        elif keyPress == "unknown":
            self.label_vector.append(0)
        self.tracker += 1

        # Create the a attribute list for the labeled segment
        if self.im_type == 'pan':
            feature_array = attr_calc.analyze_pan_image(self.original_image,
                                                        self.secondary_image,
                                                        self.im_date,
                                                        segment_id=segment_id)
        if self.im_type == 'srgb':
            feature_array = attr_calc.analyze_srgb_image(self.original_image,
                                                         self.secondary_image,
                                                         segment_id=segment_id)
        if self.im_type == 'wv02_ms':
            feature_array = attr_calc.analyze_ms_image(self.original_image,
                                                       self.secondary_image,
                                                       self.wb_ref,
                                                       self.br_ref,
                                                       segment_id=segment_id)

        # attribute_calculations returns a 2d array, but we only want the 1d list of features.
        feature_array = feature_array[0]

        live_test = True
        if live_test:
            self.print_prediction(feature_array)

        if len(self.feature_matrix) == len(self.label_vector)-1:
            #Adding all of the features found for this watershed to the main matrix
            self.feature_matrix.append(feature_array)
        else:
            old_feature_array = self.feature_matrix[len(self.label_vector)-1]
            print("Recalculated Feature.")
            print("Old: {} {}".format(old_feature_array[0],old_feature_array[1]))
            print("New: {} {}".format(feature_array[0], feature_array[1]))
            self.feature_matrix[len(self.label_vector)-1] = feature_array

        #Printing some useful statistics
        print str(self.label_vector[-1]) + ": " + keyPress

        for f in feature_array:
            print f

        # print self.original_image[self.subimage_index][self.sp_buffer[0],self.sp_buffer[1],0]
        # print feature_array
        # print "Number with Labels: %s" %len(self.label_vector)
        # print "Number with Features: %s" %len(self.feature_matrix)
        print "~"*80
        self.next_super_pixel()

    def print_prediction(self, feature_array):
        pred = self.rfc.predict(feature_array.reshape(1, -1))[0]
        pred_prob = self.rfc.predict_proba(feature_array.reshape(1, -1))[0]
        print("Predicted value: {}{}{} ({})".format(print_color.PURPLE, pred, print_color.END, pred_prob[pred]))
        return pred, pred_prob[pred]
    
    #Save progress      
    def save(self):

        print("Saving...")

        infile = h5py.File(self.savepath, 'r')
        prev_names = []
        prev_data = []
        # Compiles all of the user data that was in the previous training validation file so that
        # it can be added to the new file as well. (Because erasing and recreating a .h5 is easier
        # than altering an existing one)
        for prev_user in infile.keys():
            if prev_user != 'feature_matrix' and prev_user != 'segment_list' and prev_user != USER_NAME:
                prev_names.append(prev_user)
                prev_data.append(infile[prev_user][:])
        infile.close()

        # overwrite the h5 dataset with the updated information
        outfile = h5py.File(self.savepath, 'w')
        outfile.create_dataset('feature_matrix', data=self.feature_matrix)
        outfile.create_dataset(USER_NAME, data=self.label_vector)
        outfile.create_dataset('segment_list', data=self.segment_list)

        for i in range(len(prev_names)):
            outfile.create_dataset(prev_names[i], data=prev_data[i])

        outfile.close()
        print "Done."
    
    def exit_image(self):
        # Trim the unlabeled segments from segment list
        self.segment_list = self.segment_list[:len(self.label_vector)]
        self.save()
        self.parent.destroy()
        self.parent.quit()

    #Exits the GUI, automatically saves progress
    def quit(self):
        self.save()
        self.parent.destroy()
        self.parent.quit()
        quit()


# Returns all of the unique images in segment_list
def get_required_images(segment_list):
    image_list = []
    for seg_id in segment_list:
        if not seg_id[0] in image_list:
            image_list.append(seg_id[0])
    return image_list

def enter_username(master, entry):
    global USER_NAME
    USER_NAME = str(entry).lower()
    master.destroy()

def load_data(input_file):
    
    # the data file is the input file
    try:
        data_file = h5py.File(input_file,'r')
    except:
        print "Invalid data file."
        quit()

    # Load the existing feature matrix and segment list if they exist,
    #   otherwise initialize an empty array for these lists.
    if 'feature_matrix' in data_file.keys():
        feature_matrix = data_file['feature_matrix'][:].tolist()
    else:
        feature_matrix = []

    if 'segment_list' in data_file.keys():
        segment_list = data_file['segment_list'][:].tolist()
    else:
        # The segment list is a required entry. Setting this to false
        # triggers the appropriate warning later.
        segment_list = False

    return data_file, segment_list, feature_matrix

# Allows the user to create a new training set or classification associated with their username
# or to continue building on an existing list. 
def welcome_gui(input_file, segment_list):
    master = tk.Tk()
    master_font = tkFont.Font(family="Times New Roman", size=16)
    master['bg'] = 'white'
    master.title("Image Classifier")

    ## Exit if training_validation does not exist. This file needs to be initialized with the list of segments to classify.
    if input_file == False or segment_list == False:

        tk.Label(master, text="Could not locate necessary files, please quit and try again").grid(row=0)
        tk.Button(master, text="QUIT", command=master.destroy).grid(row=1, column=2, pady=4)
        tk.mainloop()

    tk.Label(master, text="Name:", font=master_font).grid(row=0)

    e1 = tk.Entry(master)
    e1.grid(row=0, column=1, padx=2, pady=4)
    master.bind('<Return>', lambda e: enter_username(master, e1.get()))  

    tk.Button(master, text="Enter", font=master_font, command=lambda: enter_username(master, e1.get())).grid(row=0, column=2, pady=4, padx=2)
    tk.Button(master, text="Quit", font=master_font, command=master.destroy).grid(row=1, column=2, pady=4, padx=2)

    i=0
    existing_lists = input_file.keys()
    # print existing_lists
    for key in existing_lists:
        if key == 'segment_list' or key == 'feature_matrix':
            i += 1
        else:
            number_classified = len(input_file[key][:])
            button_text = str(key) + ": " +str(number_classified)
            tk.Button(master, text=button_text, font=master_font, 
                command=lambda i=i: enter_username(master, existing_lists[i])).grid(row=i+1, column=1, pady=4)
            i += 1

    tk.mainloop()


def mode_one(segment_list, label_vector, feature_matrix, input_directory, im_type, tds_filename):
    ## Build a list of the images in the input directory
    # for each image in the directory, image list contains the "..._segmented.h5" file if it exists,
    #   and the raw image file if it does not.

    # If there is an existing segment list, note the required image to continue the
    # training set
    if segment_list:
        required_images = get_required_images(segment_list[len(label_vector):])
    else:
        required_images = []

    # Add the images in the provided folder to the image list
    image_list = []

    for ext in utils.valid_extensions:
        raw_list = utils.get_image_paths(input_directory, keyword=ext)
        for raw_im in raw_list:
            image_list.append(raw_im)

    # Save only the unique entries
    image_list = list(set(image_list))
    utils.remove_hidden(image_list)

    # Make sure we have all of the required images
    for image in required_images:
        if image in image_list:
            print "Missing required image: {}".format(image)
            quit()

    # **************************************** #
    # For expanding icebridge training set 9.18.18
    target_images = ["WV02_20140426051028_103001002F979600_14APR26051028-M1BS-500118661010_01_P002_u08rf3413_pansh_window.tif"]
    to_skip = ["WV02_20100815230724_103001000649F000_10AUG15230724-M1BS-500060589150_01_P002_u08rf3413_pansh_window.tif",
               "WV02_20140730000713_103001003405CA00_14JUL30000713-M1BS-500140639010_01_P006_u08rf3413_pansh_window.tif",
               "WV02_20150403044620_1030010040A6E100_15APR03044620-M1BS-500352729070_01_P002_u08rf3413_pansh_window.tif",
               "WV02_20150710010616_10300100464A7D00_15JUL10010616-M1BS-500518802010_01_P002_u08rf3413_pansh_window.tif",
               "WV03_20150721040829_104001000E6C2F00_15JUL21040829-M1BS-500495166010_01_P002_u08rf3413_pansh_window.tif"]
    # **************************************** #

    # As long as there were files in the input directory, loop indefinitely
    # This loop breaks when the training GUI is closed.
    while image_list:

        # Cycle through each image in the list
        for next_image in image_list:
            # Supplemental information that is stored in the training data
            im_metadata = np.zeros((17), dtype=c_int)

            # If we are out of predetermined segments to classify, start picking
            # images from those in the directory provided

            print(len(segment_list), len(label_vector))
            if len(segment_list) == len(label_vector):
                image_name = os.path.split(next_image)[1]
                if image_name in to_skip:
                    print("Skipping image: {}".format(image_name))
                    continue
                if image_name not in target_images:
                    print("Skipping {}".format(image_name))
                    continue
            # Otherwise find the image name from the next unlabeled segment
            else:
                ext = os.path.splitext(next_image)[1]
                image_name = segment_list[len(label_vector)][0] + ext
                # Convert the image name into the segmented name. This file must
                #   exist to work from an existing tds. That is, we cant resegment the
                #   image, because it will give different results and segment ids will not match.


            print("Working on image: {}".format(image_name))

            full_image_name = os.path.join(input_directory, image_name)
            src_ds = gdal.Open(full_image_name, gdal.GA_ReadOnly)

            num_bands = src_ds.RasterCount
            metadata = src_ds.GetMetadata()
            im_date = pp.parse_metadata(metadata, im_type)
            im_metadata[0] = im_date

            src_dtype = gdal.GetDataTypeSize(src_ds.GetRasterBand(1).DataType)
            lower, upper, wb_ref, br_ref = pp.histogram_threshold(src_ds, src_dtype)
            wb_ref = np.array(wb_ref, dtype=c_uint8)
            br_ref = np.array(br_ref, dtype=c_uint8)
            im_metadata[1:9] = wb_ref
            im_metadata[9:] = br_ref
            print(im_metadata)
            image_data = src_ds.ReadAsArray()
            print(lower, upper)
            image_data = pp.rescale_band(image_data, lower, upper)

            # Close the GDAL dataset
            src_ds = None

            # Rescale the input dataset using a histogram stretch and convert to the format
            #   needed by TrainingWindow
            # image_data = pp.rescale_band(image_data, lower, upper)
            # image_data = pp.white_balance(image_data, wb_reference, np.amax(wb_reference))
            # original_image = utils.create_composite([image_data[i, :, :] for i in range(0, num_bands)])
            original_image = np.ndarray.astype(image_data, c_uint8)

            print("Creating segments on provided image...")
            watershed_image = segment_image(image_data, image_type=im_type)
            # Convert the segmented image to c_int datatype. This is needed for the
            # Cython methods that calculate attribute of segments.
            watershed_image = np.ndarray.astype(watershed_image, c_uint32)

            #### Initializing the GUI
            tW = TrainingWindow(original_image, watershed_image, segment_list,
                                image_name, label_vector, feature_matrix, im_type,
                                im_metadata, tds_filename)

            # Reload the datafiles after the window closes
            data_file, segment_list, feature_matrix = load_data(tds_filename)
            label_vector = data_file[USER_NAME][:].tolist()
            data_file.close()


def main():
    
    #### Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="folder containing training images")
    parser.add_argument("image_type", type=str, choices=['srgb','wv02_ms','pan'],
                        help="image type: 'srgb', 'wv02_ms', 'pan'")
    parser.add_argument("--tds_file", type=str, default=None,
                        help='''Existing training dataset file. Will create a new one with this name if none exists.
                        default: <image_type>_training_data.h5''')

    #### Parse Arguments
    args = parser.parse_args()
    im_type = args.image_type
    tds_file = args.tds_file

    input_directory = args.input
    # Set the default tds filename if this was not entered
    if tds_file == None:
        tds_file = im_type + "_training_data.h5"
    # Force the extension no matter what the user entered
    else:
        tds_file = os.path.splitext(tds_file)[0] + ".h5"

    # If the user input a valid file, then split the filename and path
    if os.path.isfile(tds_file):
        # tds_directory, tds_file = os.path.split(tds_file)
        input_file = tds_file
    elif os.path.isfile(os.path.join(input_directory,tds_file)):
        input_file = os.path.join(input_directory,tds_file)
    # Otherwise create a dummy file
    else:
        input_file = os.path.join(input_directory,tds_file)
        temp_file = h5py.File(input_file,'w')
        temp_file.create_dataset("segment_list",data=[])
        temp_file.create_dataset("feature_matrix",data=[])
        temp_file.create_dataset(im_type, data=[])
        temp_file.close()

    # input_path is the path of either the training dataset or the accuracy
    # input_path = os.path.dirname(input_directory)
    tds_filename = os.path.join(input_directory, tds_file)

    #### Load Necessary Files
    # Loads the files to use for manual classification.
    # Mode 1 requires:
    #   tds.h5 file, which contains the list of segments to be classified, and any work already done.
    # Mode 2 requires:
    #  the accuracy data file, which only has previous work done for that image.

    # NOTE: In mode 2, segment_list is the ID of manually classified pixels,
    # and feature_matrix is a simple list (not matrix) of the predicted classifications
    data_file, segment_list, feature_matrix = load_data(input_file)

    # Run a gui for selecting username for training set or classification list creation
    # This method creates the global var USER_NAME
    welcome_gui(data_file, segment_list)

    # Determine if this user has data already stored in the training set. If so,
    # use the existing classifications. If not, start from the beginning. 
    # must use .tolist() because datasets in h5py files are numpy arrays, and we want
    # these as python lists. 
    try: 
        if USER_NAME in data_file.keys():
            label_vector = data_file[USER_NAME][:].tolist()
            # print "Username found: You have classified %s segments so far." %len(label_vector)
        else:
            # print "Username not found: Starting classifications from beginning."
            label_vector = [] # [y1...yn] column vector where n : number of classified segments, y = classification
    except NameError:
        data_file.close()
        print("Name Error")
        quit()

    data_file.close()

    #### Load the images and segments.
    mode_one(segment_list, label_vector, feature_matrix, input_directory, im_type, tds_filename)



if __name__ == "__main__":
    main()
