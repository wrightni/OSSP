#title: Training Set Creation for Random Forest Classification
#author: Nick Wright
#Inspired by: Justin Chen

#purpose: Creates a GUI for a user to identify watershed superpixels of an image as
#        melt ponds, sea ice, or open water to use as a training data set for a 
#        Random Forest Classification method. 

# Python 3:
import tkinter as tk
# Python 2:
# import Tkinter as tk
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


class PrintColor:
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


class Buttons(tk.Frame):
    # Defines the properties of all the controller buttons to be used by the GUI.
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        prev_btn = tk.Button(self, text="Previous Segment", width=16, height=2,
                             command=lambda: parent.event_manager.previous_segment())
        prev_btn.grid(column=0, row=0, pady=(0,20))

        water_btn = tk.Button(self, text="Open Water", width=16, height=2, highlightbackground='#000000',
                              command=lambda: parent.event_manager.classify("water"))
        water_btn.grid(column=0, row=1, pady=1)

        melt_btn = tk.Button(self, text="Melt Pond", width=16, height=2, highlightbackground='#4C678C',
                             command=lambda: parent.event_manager.classify("melt"))
        melt_btn.grid(column=0, row=2, pady=1)

        gray_btn = tk.Button(self, text="Dark and Thin Ice", width=16, height=2, highlightbackground='#D2D3D5',
                             command=lambda: parent.event_manager.classify("gray"))
        gray_btn.grid(column=0, row=3, pady=1)

        snow_btn = tk.Button(self, text="Snow or Ice", width=16, height=2,
                             command=lambda: parent.event_manager.classify("snow"))
        snow_btn.grid(column=0, row=4, pady=1)

        shadow_btn = tk.Button(self, text="Shadow", width=16, height=2, highlightbackground='#FF9200',
                               command=lambda: parent.event_manager.classify("shadow"))
        shadow_btn.grid(column=0, row=5, pady=1)

        unknown_btn = tk.Button(self, text="Unknown / Mixed", width=16, height=2,
                                command=lambda: parent.event_manager.classify("unknown"))
        unknown_btn.grid(column=0, row=6, pady=1)

        auto_btn = tk.Button(self, text="Autorun", width=16, height=2,
                             command=lambda: parent.event_manager.autorun())
        auto_btn.grid(column=0, row=7, pady=(20,0))

        next_btn = tk.Button(self, text="Next Image", width=16, height=2,
                             command=lambda: parent.event_manager.next_image())
        next_btn.grid(column=0, row=8, pady=1)
        quit_btn = tk.Button(self, text="Save and Quit", width=16, height=2,
                             command=lambda: parent.event_manager.quit_event())
        quit_btn.grid(column=0, row=9, pady=1)

        load_first_btn = tk.Button(self, text="Initialize Image", width=16, height=2,
                                   command=lambda: parent.event_manager.initialize_image())
        load_first_btn.grid(column=0, row=10, pady=(40,0))


class ProgressBar(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        self.total_counter = tk.StringVar()
        self.total_counter.set("Total Progress: {}".format(0))
        self.image_tracker = tk.StringVar()
        self.image_tracker.set("")

        total_text = tk.Label(self, textvariable=self.total_counter)
        total_text.grid(column=0, row=0)

        image_text = tk.Label(self, textvariable=self.image_tracker)
        image_text.grid(column=0, row=1)

    def update_progress(self):
        self.total_counter.set("Total Progress: {}".format(self.parent.data.get_num_labels()))
        self.image_tracker.set("Image {} of {}".format(self.parent.data.im_index + 1,
                                                       len(self.parent.data.available_images)))


class ImageDisplay(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        # Initialize class variables
        # Populated in initialize_image method:
        self.display_image = None
        self.disp_xdim, self.disp_ydim, = 0, 0
        # Populated in update_images:
        self.zoom_win_x, self.zoom_win_y = 0, 0

        # Creating the canvas where the images will be
        self.fig = plt.figure(figsize=[10, 10])
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.99, wspace=0.01, hspace=0.01)

        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.draw()
        # toolbar = NavigationToolbar2TkAgg(canvas, frame)
        canvas.get_tk_widget().grid(column=0, row=0)
        # toolbar.pack(in_=frame, side='top')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', parent.event_manager.onclick)

        # Create a placeholder while image data is loading
        self.initial_display()


    def initialize_image(self):
        # Creates a local composite of the original image data for display
        if self.parent.data.im_type == 'wv02_ms':
            self.display_image = utils.create_composite([self.parent.data.original_image[4, :, :],
                                                         self.parent.data.original_image[2, :, :],
                                                         self.parent.data.original_image[1, :, :]],
                                                         dtype=np.uint8)

        elif self.parent.data.im_type == 'pan':
            self.display_image = utils.create_composite([self.parent.data.original_image,
                                                         self.parent.data.original_image,
                                                         self.parent.data.original_image],
                                                         dtype=np.uint8)

        elif self.parent.data.im_type == 'srgb':
            self.display_image = utils.create_composite([self.parent.data.original_image[0, :, :],
                                                         self.parent.data.original_image[1, :, :],
                                                         self.parent.data.original_image[2, :, :]],
                                                         dtype=np.uint8)
        self.disp_xdim, self.disp_ydim = np.shape(self.display_image)[0:2]


    def loading_display(self):

        plt.clf()

        loading_text = "Images are loading, please wait... "
        # Creates a image placeholder while the data is being loaded.
        ax = self.fig.add_subplot(1, 1, 1, adjustable='datalim', frame_on=False)
        ax.text(0.5, 0.5, loading_text, horizontalalignment='center', verticalalignment='center')
        ax.axis('off')

        # Updating the plots
        self.fig.canvas.draw()

    def initial_display(self):

        plt.clf()

        welcome_text = "No images have been loaded. Press <Initialize Image> to begin."
        tds_text = "Training data file: \n {}".format(self.parent.data.tds_filename)
        image_text = "Images found: \n"
        if len(self.parent.data.available_images) == 0:
            image_text += 'None'
        else:
            for im in self.parent.data.available_images:
                image_text += im + '\n'

        # Creates a image placeholder while the data is being loaded.
        ax = self.fig.add_subplot(2, 1, 1, adjustable='datalim', frame_on=False)
        ax.text(0.5, 0.3, welcome_text, horizontalalignment='center', verticalalignment='bottom', weight='bold')
        ax.axis('off')

        ax2 = self.fig.add_subplot(2, 1, 2, adjustable='datalim', frame_on=False)
        ax2.text(0.5, 1, tds_text, horizontalalignment='center', verticalalignment='center')
        ax2.text(0.5, .9, image_text, horizontalalignment='center', verticalalignment='top')
        ax2.axis('off')

        # Updating the plots
        self.fig.canvas.draw()

    def update_images(self, segment_id):
        # Clear the existing display
        plt.clf()

        current_seg = self.parent.data.segmented_image == segment_id   # array of 0 or 1 where 1 = current segment
        segment_pos = np.nonzero(current_seg)                # returns the array position of the segment

        zoom_size = 100

        x_min = np.amin(segment_pos[0]) - zoom_size
        x_max = np.amax(segment_pos[0]) + zoom_size
        y_min = np.amin(segment_pos[1]) - zoom_size
        y_max = np.amax(segment_pos[1]) + zoom_size

        # Store the zoom window corner coordinates for reference in onclick()
        # xMin and yMin are defined backwards
        self.zoom_win_x = y_min
        self.zoom_win_y = x_min

        if x_min < 0:
            x_min = 0
        if x_max >= self.disp_xdim:
            x_max = self.disp_xdim - 1
        if y_min < 0:
            y_min = 0
        if y_max >= self.disp_ydim:
            y_max = self.disp_ydim - 1

        # Image 2 (Zoomed in image, no highlighted segment)
        cropped_image = self.display_image[x_min:x_max, y_min:y_max]

        # Image 3 (Zoomed in image, with segment highlight)
        color_image = np.copy(self.display_image)
        color_image[:, :, 0][current_seg] = 255
        color_image[:, :, 2][current_seg] = 0
        color_image = color_image[x_min:x_max, y_min:y_max]

        # Text instructions
        instructions = '''
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
        '''

        # Plotting onto the GUI
        ax = self.fig.add_subplot(2, 2, 1)
        ax.imshow(color_image, interpolation='None', vmin=0, vmax=255)
        ax.tick_params(axis='both',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom=False,  # ticks along the bottom edge are off
                       top=False,  # ticks along the top edge are off
                       left=False,
                       right=False,
                       labelleft=False,
                       labelbottom=False)
        ax.set_label('ax1')

        ax = self.fig.add_subplot(2, 2, 2)
        ax.imshow(cropped_image, interpolation='None', vmin=0, vmax=255)
        ax.tick_params(axis='both',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom=False,  # ticks along the bottom edge are off
                       top=False,  # ticks along the top edge are off
                       left=False,
                       right=False,
                       labelleft=False,
                       labelbottom=False)
        ax.set_label('ax2')

        ax = self.fig.add_subplot(2, 2, 3)
        ax.imshow(self.display_image, interpolation='None', vmin=0, vmax=255)
        ax.axvspan(y_min,
                   y_max,
                   1. - float(x_max) / self.disp_xdim,
                   1. - float(x_min) / self.disp_xdim,
                   color='red',
                   alpha=0.3)
        ax.set_xlim([0, np.shape(self.display_image)[1]])
        ax.tick_params(axis='both',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom=False,  # ticks along the bottom edge are off
                       top=False,  # ticks along the top edge are off
                       left=False,
                       right=False,
                       labelleft=False,
                       labelbottom=False)
        ax.set_label('ax3')

        ax = self.fig.add_subplot(2, 2, 4, adjustable='datalim', frame_on=False)
        ax.text(0.5, 0.5, instructions, horizontalalignment='center', verticalalignment='center')
        ax.axis('off')

        # Updating the plots
        self.fig.canvas.draw()


class DataManager:

    def __init__(self, available_images, tds_filename, username, im_type):

        # Image and segment data (populated in load_image())
        self.original_image = None
        self.segmented_image = None

        # Variable Values   (populated in load_training_data())
        self.label_vector = []
        self.segment_list = []
        self.feature_matrix = []
        self.tracker = 0                                    # Number of segment sets added from the current image
        self.im_index = 0                                   # Index for progressing through available images

        # Global Static Values
        self.tds_filename = tds_filename
        self.username = username
        self.im_type = im_type
        self.available_images = available_images

        # Image Static Value (populated in load_image())
        self.wb_ref = None
        self.br_ref = None
        self.im_date = None
        self.im_name = None

    def load_next_image(self):
        # Increment the image index
        self.im_index += 1
        # Loop im_index based on the available number of images
        self.im_index = self.im_index % len(self.available_images)
        # Load the new data
        self._load_image()

    def load_previous_image(self):
        # If an image has already been loaded, and there is no previous data,
        #   prevent the user from using this button.
        if self.get_num_labels() == 0 and self.im_name is not None:
            return

        # If labels exist find the correct image to load
        if self.get_num_labels() != 0:
            # If this does not find a match, im_index will default to its current value
            for i in range(len(self.available_images)):
                if self.get_current_segment()[0] in self.available_images[i]:
                    self.im_index = i

        self._load_image()

    def _load_image(self):
        # Loads the optical and segmented image data from disk. Should only be called from
        #   load_next_image method.
        full_image_name = self.available_images[self.im_index]

        self.im_name = os.path.splitext(os.path.split(full_image_name)[1])[0]

        src_ds = gdal.Open(full_image_name, gdal.GA_ReadOnly)

        # Read the image date from the metadata
        metadata = src_ds.GetMetadata()
        self.im_date = pp.parse_metadata(metadata, self.im_type)

        # Determine the datatype
        src_dtype = gdal.GetDataTypeSize(src_ds.GetRasterBand(1).DataType)

        # Calculate the reference points from the image histogram
        lower, upper, wb_ref, br_ref = pp.histogram_threshold(src_ds, src_dtype)
        self.wb_ref = np.array(wb_ref, dtype=c_uint8)
        self.br_ref = np.array(br_ref, dtype=c_uint8)

        # Load the image data
        image_data = src_ds.ReadAsArray()

        # Close the GDAL dataset
        src_ds = None

        # Rescale the input dataset using a histogram stretch
        image_data = pp.rescale_band(image_data, lower, upper)

        # Apply a white balance to the image
        image_data = pp.white_balance(image_data, self.wb_ref.astype(np.float), float(np.amax(self.wb_ref)))

        # Convert the input data to c_uint8
        self.original_image = np.ndarray.astype(image_data, c_uint8)

        print("Creating segments on provided image...")
        watershed_image = segment_image(image_data, image_type=self.im_type)
        # Convert the segmented image to c_int datatype. This is needed for the
        # Cython methods that calculate attribute of segments.
        self.segmented_image = np.ndarray.astype(watershed_image, c_uint32)
        # Clear these from memory explicitly
        image_data = None
        watershed_image = None

    def load_training_data(self):

        try:
            with h5py.File(self.tds_filename, 'r') as data_file:
                # Load the existing feature matrix and segment list if they exist,
                #   otherwise initialize an empty array for these lists.
                if 'feature_matrix' in list(data_file.keys()):
                    self.feature_matrix = data_file['feature_matrix'][:].tolist()
                else:
                    self.feature_matrix = []

                if 'segment_list' in list(data_file.keys()):
                    # For loading files created in py2
                    self.segment_list = [[name[0].decode(), name[1].decode()] for name in data_file['segment_list']]
                else:
                    self.segment_list = []

                # Determine if this user has data already stored in the training set. If so,
                # use the existing classifications. If not, start from the beginning.
                # must use .tolist() because datasets in h5py files are numpy arrays, and we want
                # these as python lists.
                # [y1...yn] column vector where n : number of classified segments, y = classification
                if self.username in list(data_file.keys()):
                    self.label_vector = data_file[self.username][:].tolist()
                else:
                    self.label_vector = []
        # If the file does not exist, create empty values
        except OSError:
            self.feature_matrix = []
            self.segment_list = []
            self.label_vector = []

    def get_num_labels(self):
        return len(self.label_vector)

    def append_label(self, label):
        self.tracker += 1
        self.label_vector.append(label)

    # Removes the last entry from label_vector
    def remove_last_label(self):
        self.label_vector.pop()
        self.tracker -= 1

    def get_num_segments(self):
        return len(self.segment_list)

    # The current segment is the next one that doesn't have an associated label
    def get_current_segment(self):
        return self.segment_list[len(self.label_vector)]

    def add_single_segment(self, new_segment):
        self.segment_list.append(new_segment)

    # Trims all unclassified segments from segment_list by trimming it to
    # the length of label_vector
    def trim_segment_list(self):
        self.segment_list = self.segment_list[:len(self.label_vector)]

    # Add 10 randomly selected segments to the list of ones to classify
    def add_segments(self):
        segments_to_add = []

        a = 0
        # Select random x,y coordinates from the input image, and pick the segment where the random
        #   pixel lands. This makes the selected segments representative of the average surface
        #   distribution within the image. This still wont work if the image has a minority of any
        #   particular surface type.
        while len(segments_to_add)<10:
            a += 1
            z, x, y = np.shape(self.original_image)
            i = np.random.randint(x)
            j = np.random.randint(y)
            # Find the segment label at the random pixel
            segment_id = self.segmented_image[i][j]
            sp_size = np.sum(self.segmented_image == segment_id)
            if sp_size >= 20:
                # Check for a duplicate segment already in the tds
                new_segment = [self.im_name,
                               "{}".format(segment_id)]
                if new_segment not in self.segment_list and new_segment not in segments_to_add:
                    segments_to_add.append(new_segment)

        print(("Attempts: {}".format(a)))
        self.segment_list += segments_to_add

    def compute_attributes(self, segment_id):
        # Create the a attribute list for the labeled segment
        feature_array = calc_attributes(self.original_image, self.segmented_image,
                                        self.wb_ref, self.br_ref, self.im_date, segment_id, self.im_type)

        # attribute_calculations returns a 2d array, but we only want the 1d list of features.
        feature_array = feature_array[0]

        return feature_array

    def append_features(self, feature_array):
        # If there are fewer features than labels, assume the new one should be appended
        # to the end
        if len(self.feature_matrix) == len(self.label_vector) - 1:
            #Adding all of the features found for this watershed to the main matrix
            self.feature_matrix.append(feature_array)
        # Otherwise replace the existing features with the newly calculated ones.
        # (Maybe just skip this in the future and assume they were calculated correctly before?
        else:
            # old_feature_array = self.feature_matrix[len(self.label_vector) - 1]
            print("Recalculated Feature.")
            # print(("Old: {} {}".format(old_feature_array[0], old_feature_array[1])))
            # print(("New: {} {}".format(feature_array[0], feature_array[1])))
            self.feature_matrix[len(self.label_vector) - 1] = feature_array


class EventManager:

    def __init__(self, parent):
        self.parent = parent
        self.is_active = False                  # Prevents events from happening while images are loading

    def activate(self):
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def next_segment(self):
        if not self.is_active:
            return

        # If all of the segments in the predefined list have been classified already,
        # present the user with a random new segment.
        if self.parent.data.get_num_labels() == self.parent.data.get_num_segments():

            # I think if segment_list == [] is covered by the above..?

            self.parent.data.add_segments()

            # retrain the random forest model if the live predictor is active
            if self.parent.live_predictor.is_active():
                self.parent.live_predictor.retrain_model(self.parent.data.feature_matrix,
                                                         self.parent.data.label_vector)

        # The current segment is the next one that doesn't have an associated label
        current_segment = self.parent.data.get_current_segment()
        segment_id = int(current_segment[1])

        # Redraw the display with the new segment id
        self.parent.image_display.update_images(segment_id)

    def previous_segment(self):
        if not self.is_active:
            return
        # Make sure this function returns null if there is no previous sp to go back to
        if self.parent.data.get_num_labels() == 0:
            return
        else:
            # Delete the last label in the list, then get the 'new' current segment
            self.parent.data.remove_last_label()
            current_segment = self.parent.data.get_current_segment()
            self.parent.progress_bar.update_progress()

            if current_segment[0] != self.parent.data.im_name:
                self.previous_image()
                return

            segment_id = int(current_segment[1])
            # Redraw the display with the new segment id
            self.parent.image_display.update_images(segment_id)

    def onclick(self, event):
        if not self.is_active:
            return

        if event.inaxes is not None:
            axes_properties = event.inaxes.properties()
            segment_id = -1
            x, y = 0, 0

            # If the mouse click was in the overview image
            if axes_properties['label'] == 'ax3':
                x = int(event.xdata)
                y = int(event.ydata)
                segment_id = self.parent.data.segmented_image[y, x]

            # Either of the top zoomed windows
            if axes_properties['label'] == 'ax1' or axes_properties['label'] == 'ax2':
                win_x = int(event.xdata)
                win_y = int(event.ydata)
                x = self.parent.image_display.zoom_win_x + win_x
                y = self.parent.image_display.zoom_win_y + win_y
                segment_id = self.parent.data.segmented_image[y, x]

            # If user clicked on a valid location, add the segment that was clicked on to segment_list,
            #   then update the image render.
            if segment_id >= 0:
                print(("You clicked at ({}, {}) in {}".format(x, y, axes_properties['label'])))
                print(("Segment id: {}".format(segment_id)))
                new_segment = [self.parent.data.im_name,
                               "{}".format(segment_id)]
                if new_segment not in self.parent.data.segment_list:
                    # Trim all unclassified segments
                    self.parent.data.trim_segment_list()
                    # Add the selected one as the next segment
                    self.parent.data.add_single_segment(new_segment)
                    # Get the new current segment and redraw display
                    segment_id = int(self.parent.data.get_current_segment()[1])
                    self.parent.image_display.update_images(segment_id)
                else:
                    print("This segment has already been labeled")

    def classify(self, key_press):
        if not self.is_active:
            return

        # Assigning the highlighted segment a classification
        segment_id = int(self.parent.data.get_current_segment()[1])
        print("Segment ID: {}".format(segment_id))
        # Note that we classified one more image
        if key_press == "snow":
            self.parent.data.append_label(1)
        elif key_press == "gray":
            self.parent.data.append_label(2)
        elif key_press == "melt":
            self.parent.data.append_label(3)
        elif key_press == "water":
            self.parent.data.append_label(4)
        elif key_press == "shadow":
            self.parent.data.append_label(5)
        elif key_press == "unknown":
            self.parent.data.append_label(6)

        # Calculate the attributes for the current segment
        feature_array = self.parent.data.compute_attributes(segment_id)
        self.parent.data.append_features(feature_array)

        # Printing some useful statistics
        print("Assigned value: {} ({})".format(str(self.parent.data.label_vector[-1]), key_press))

        if self.parent.live_predictor.is_active():
            self.parent.live_predictor.print_prediction(feature_array)

        print(("~"*80))

        self.parent.progress_bar.update_progress()

        self.next_segment()

        # if len(self.feature_matrix) == len(self.label_vector)-1:
        #     #Adding all of the features found for this watershed to the main matrix
        #     self.feature_matrix.append(feature_array)
        # else:
        #     old_feature_array = self.feature_matrix[len(self.label_vector)-1]
        #     print("Recalculated Feature.")
        #     print(("Old: {} {}".format(old_feature_array[0],old_feature_array[1])))
        #     print(("New: {} {}".format(feature_array[0], feature_array[1])))
        #     self.feature_matrix[len(self.label_vector)-1] = feature_array

    def autorun(self):
        if not self.is_active:
            return

        # In the future make this function a standalone window (instead of terminal output)??
        # Prevent the user from accessing this if the predictor is inactive
        if not self.parent.live_predictor.is_active():
            print("Autorun functionality disabled")
            return

        # segment_id = int(self.segment_list[len(self.label_vector):][0][1])
        segment_id = int(self.parent.data.get_current_segment()[1])

        # Create the a attribute list for the labeled segment
        feature_array = self.parent.data.compute_attributes(segment_id)

        # feature_array = calc_attributes(self.original_image, self.secondary_image,
        #                                 self.wb_ref, self.br_ref, self.im_date, segment_id, self.im_type)

        print("~" * 80)
        # This both prints the results of the prediction for the user to check, and also returns the
        #   predicted values for use here.
        pred, proba = self.parent.live_predictor.print_prediction(feature_array)
        if 0.90 < proba < 0.96:
            timeout = 4 #6
            print((PrintColor.BOLD + "Label if incorrect:" + PrintColor.END))
        elif proba < .9:
            timeout = 10 #12
            print((PrintColor.BOLD + PrintColor.RED + "Label if incorrect:" + PrintColor.END))
        else:
            timeout = 0.5

        # Prompt the user to change the classification if they dont agree with the
        #   predicted one. If no input is recieved, the predicted one is assumed to be correct.
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
                print(("Assigning label {} instead.".format(label)))
            else:
                print("Ending autorun.")
                return
        else:
            label = pred
            print(("No input. Assigning label: {}".format(label)))

        self.parent.data.append_label(label)
        self.parent.data.append_features(feature_array)

        self.parent.progress_bar.update_progress()

        self.next_segment()
        self.parent.after(100, self.autorun)

    def save(self):

        if self.parent.data.label_vector == []:
            return

        print("Saving...")

        username = self.parent.data.username

        prev_names = []
        prev_data = []
        try:
            with h5py.File(self.parent.data.tds_filename, 'r') as infile:
                # Compiles all of the user data that was in the previous training validation file so that
                # it can be added to the new file as well. (Because erasing and recreating a .h5 is easier
                # than altering an existing one)
                for prev_user in list(infile.keys()):
                    if prev_user != 'feature_matrix' and prev_user != 'segment_list' and prev_user != username:
                        prev_names.append(prev_user)
                        prev_data.append(infile[prev_user][:])
                infile.close()
        except OSError:
            pass

        # overwrite the h5 dataset with the updated information
        with h5py.File(self.parent.data.tds_filename, 'w') as outfile:
            outfile.create_dataset('feature_matrix', data=self.parent.data.feature_matrix)
            outfile.create_dataset(username, data=self.parent.data.label_vector)
            segment_list = np.array(self.parent.data.segment_list, dtype=np.string_)
            outfile.create_dataset('segment_list', data=segment_list)

            for i in range(len(prev_names)):
                outfile.create_dataset(prev_names[i], data=prev_data[i])

        print("Done.")

    def next_image(self):
        if not self.is_active:
            return

        self.deactivate()
        # Trim the unlabeled segments from segment list
        self.parent.data.trim_segment_list()
        # Save the existing data
        self.save()
        # Set the display to the loading screen
        self.parent.after(10, self.parent.image_display.loading_display())
        # Load the next image data
        self.parent.data.load_next_image()
        # Add the new data to the display class
        self.parent.image_display.initialize_image()
        # Update the display screen
        # Go to the next segment (which will add additional segments to the queue and update the display)
        self.parent.progress_bar.update_progress()
        self.activate()
        self.next_segment()

    def previous_image(self):

        self.deactivate()
        # Set the display to the loading screen
        self.parent.after(10, self.parent.image_display.loading_display())
        # Load the previous image data
        self.parent.data.load_previous_image()
        # Add the new data to the display class
        self.parent.image_display.initialize_image()
        # Update the display screen
        # Go to the next segment (which will add additional segments to the queue and update the display)
        self.parent.progress_bar.update_progress()
        self.activate()
        self.next_segment()

    def initialize_image(self):
        if len(self.parent.data.available_images) == 0:
            print("No images to load!")
            return
        # Check to make sure no data has been loaded
        if self.parent.data.im_name is not None:
            return
        # Previous image does all the loading work we need for the first image
        self.previous_image()


    def quit_event(self):
        # Exits the GUI, automatically saves progress
        self.save()
        self.parent.exit_gui()


class LivePredictor:

    def __init__(self, active_state):
        self.active_state = active_state
        self.is_trained = False
        self.rfc = RandomForestClassifier(n_estimators=100)

    # True if LivePredictor is running, false otherwise
    def is_active(self):
        return self.active_state

    def retrain_model(self, feature_matrix, label_vector):
        if len(label_vector) >= 10:
            self.rfc.fit(feature_matrix[:len(label_vector)], label_vector)
            self.is_trained = True

    def print_prediction(self, feature_array):
        if self.is_trained:
            pred = self.rfc.predict(feature_array.reshape(1, -1))[0]
            pred_prob = self.rfc.predict_proba(feature_array.reshape(1, -1))[0]
            pred_prob = np.amax(pred_prob)
            print(("Predicted value: {}{}{} ({})".format(PrintColor.PURPLE, pred, PrintColor.END, pred_prob)))
            return pred, pred_prob
        else:
            return 0, 0


class TrainingWindow(tk.Frame):

    def __init__(self, parent, img_list, tds_filename, username, im_type, activate_autorun=False):

        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Training GUI")

        # Create the controlling buttons and place them on the right side.
        self.buttons = Buttons(self)
        self.buttons.grid(column=1, row=1, sticky="N")

        # Manager for all the GUI events (e.g. button presses)
        self.event_manager = EventManager(self)

        # Data manager object
        self.data = DataManager(img_list, tds_filename, username, im_type)
        self.data.load_training_data()

        # Create the image display window
        self.image_display = ImageDisplay(self)
        self.image_display.grid(column=0, row=0, rowspan=2)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.grid(column=1, row=0)

        self.progress_bar.update_progress()

        # Object for creating on the fly predictions and managing the auto_run method
        self.live_predictor = LivePredictor(activate_autorun)

        # Define keybindings
        self.parent.bind('1', lambda e: self.event_manager.classify("snow"))
        self.parent.bind('2', lambda e: self.event_manager.classify("gray"))
        self.parent.bind('3', lambda e: self.event_manager.classify("melt"))
        self.parent.bind('4', lambda e: self.event_manager.classify("water"))
        self.parent.bind('5', lambda e: self.event_manager.classify("shadow"))
        self.parent.bind('<Tab>', lambda e: self.event_manager.classify("unknown"))
        self.parent.bind('<BackSpace>', lambda e: self.event_manager.previous_segment())


    def exit_gui(self):
        self.parent.quit()
        self.parent.destroy()


def calc_attributes(original_image, secondary_image,
                    wb_ref, br_ref, im_date, segment_id, im_type):
    feature_array = []

    if im_type == 'pan':
        feature_array = attr_calc.analyze_pan_image(original_image,
                                                    secondary_image,
                                                    im_date,
                                                    segment_id=segment_id)
    if im_type == 'srgb':
        feature_array = attr_calc.analyze_srgb_image(original_image,
                                                     secondary_image,
                                                     segment_id=segment_id)
    if im_type == 'wv02_ms':
        feature_array = attr_calc.analyze_ms_image(original_image,
                                                   secondary_image,
                                                   wb_ref,
                                                   br_ref,
                                                   segment_id=segment_id)
    return feature_array


# Returns all of the unique images in segment_list
def get_required_images(segment_list):
    image_list = []
    for seg_id in segment_list:
        if not seg_id[0] in image_list:
            image_list.append(seg_id[0])
    return image_list


def validate_tds_file(tds_filename, input_dir, image_type):

    # Set the default tds filename if this was not entered
    if tds_filename is None:
        tds_filename = os.path.join(input_dir, image_type + "_training_data.h5")
    elif os.path.isfile(tds_filename):
        # If a real file was given, try opening it.
        try:
            data_file = h5py.File(tds_filename, 'r')
            data_file.close()
        except OSError:
            print("Invalid data file.")
            quit()

    return tds_filename


# Finds all the unique images from the given directory
def scrape_dir(src_dir):
    image_list = []

    for ext in utils.valid_extensions:
        raw_list = utils.get_image_paths(src_dir, keyword=ext)
        for raw_im in raw_list:
            image_list.append(raw_im)

    # Save only the unique entries
    image_list = list(set(image_list))
    utils.remove_hidden(image_list)

    return image_list


if __name__ == "__main__":
    
    #### Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="folder containing training images")
    parser.add_argument("image_type", type=str, choices=['srgb','wv02_ms','pan'],
                        help="image type: 'srgb', 'wv02_ms', 'pan'")
    parser.add_argument("--tds_file", type=str, default=None,
                        help='''Existing training dataset file. Will create a new one with this name if none exists.
                        default: <image_type>_training_data.h5''')
    parser.add_argument("--username", type=str, default=None,
                        help='''username to associate with the training set.
                             default: image_type''')
    parser.add_argument("-a", "--enable_autorun", action="store_true",
                        help='''Enables the use of the autorun function.''')

    # Parse Arguments
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input)
    image_type = args.image_type
    autorun_flag = args.enable_autorun

    # Add the images in the provided folder to the image list
    img_list = scrape_dir(input_dir)

    tds_file = validate_tds_file(args.tds_file, input_dir, image_type)

    if args.username is None:
        user_name = image_type
    else:
        user_name = args.username

    root = tk.Tk()
    TrainingWindow(root, img_list, tds_file, user_name, image_type,
                   activate_autorun=autorun_flag).pack(side='top', fill='both', expand=True)
    root.mainloop()
