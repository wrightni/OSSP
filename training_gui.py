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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from scipy.misc import bytescale
import h5py
import os
import argparse
from skimage.filters.rank import entropy
from skimage.morphology import disk

from preprocess import prepare_image
from segment import segment_image, load_from_disk
from lib import utils, feature_calculations


#This class structure logic might be a little far-fetched
class TrainingWindow:

    def __init__(self, original_image, secondary_image, segment_list, image_name, label_vector, feature_matrix, 
        im_type, im_date, mode, savepath):
        """Constructor"""
        #Variables
        self.parent = tk.Tk()
        self.original_image = original_image
        # In mode 1, secondary image is the watershed segments
        # In mode 2, secondary image is the classified image
        self.secondary_image = secondary_image
        self.label_vector = label_vector
        self.feature_matrix = feature_matrix
        self.image_name = str(image_name)
        # Format of the ID values in segment_list
        # 1: ["image_name", subimage, segment_number]
        # 2: [subimage,x,y]
        self.segment_list = segment_list
        self.im_type = im_type                  # Input image type
        self.im_date = im_date
        self.mode = mode                        # 1: training creation | 2: accuracy assessment
        self.savepath = savepath
        self.tracker = 0                        # Number of segment sets added from the current image
        # If there is no existing segment list or the segment list and label list are the same length
        #  initialize the subimage index and pixel buffer with random integers. 
        if segment_list == [] or len(label_vector) == len(segment_list):
            # Find some segments to add to the 'to-do' list
            self.add_segments()
            # Fill sp_buffer with the first segment's info
            self.sp_buffer = segment_list[len(label_vector)][2]
            # Find the subimage index
            self.subimage_index = int(segment_list[len(label_vector)][1])
        # Pull the first unclassified segment from segment list, then parse the needed number from that id
        # based on the mode we are using.
        elif mode == 1:
            self.sp_buffer = segment_list[len(label_vector)][2]   # current super pixel
            self.subimage_index = int(segment_list[len(label_vector)][1])
        elif mode == 2:
            self.sp_buffer = (segment_list[len(label_vector)][1],segment_list[len(label_vector)][2])
            self.subimage_index = int(segment_list[len(label_vector)][0])
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
        shadowBtn = tk.Button(self.parent, text="Shadow", width=16, height=2, highlightbackground='#FF9200',
            command=lambda: self.classify("shadow"))
        shadowBtn.pack(in_=buttons, side='top')
        unknownBtn = tk.Button(self.parent, text="Unknown / Mixed", width=16, height=2, command=lambda: self.classify("unknown"))
        unknownBtn.pack(in_=buttons, side='top')
        spacer2 = tk.Button(self.parent, width=16, height=2)
        spacer2.pack(in_=buttons,side="top")
        quitBtn = tk.Button(self.parent, text="Save and Quit", width=16, height=2, command=lambda: self.quit())
        quitBtn.pack(in_=buttons, side='top')

        self.parent.bind('1', lambda e: self.classify("snow")) 
        self.parent.bind('2', lambda e: self.classify("gray")) 
        self.parent.bind('3', lambda e: self.classify("melt")) 
        self.parent.bind('4', lambda e: self.classify("water")) 
        self.parent.bind('5', lambda e: self.classify("shadow"))
        self.parent.bind('<Tab>', lambda e: self.classify("unknown"))
        self.parent.bind('<BackSpace>', lambda e: self.previous_super_pixel())


        #Creating the canvas where the images will be
        self.fig = plt.figure(figsize=[10,10])
        self.fig.subplots_adjust(left=0.01,right=0.99,bottom=0.05,top=0.99,wspace=0.01,hspace=0.01)
        canvas = FigureCanvasTkAgg(self.fig, frame)
        # toolbar = NavigationToolbar2TkAgg(canvas, frame)
        canvas.get_tk_widget().pack(in_=frame, side='top')
        # toolbar.pack(in_=frame, side='top')
        self.next_super_pixel()
        self.parent.mainloop()
    
    #Displays the 3 images
    def displayImages(self):
    
        plt.clf()

        if self.im_type == 'wv02_ms':
            band_1 = self.original_image[self.subimage_index][:,:,0] #band 1
            img_dim = band_1.shape
            subimage = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.uint8)    
            subimage[:,:,0] = self.original_image[self.subimage_index][:,:,4] #band 5 (red)
            subimage[:,:,1] = self.original_image[self.subimage_index][:,:,2] #band 3 (green)
            subimage[:,:,2] = self.original_image[self.subimage_index][:,:,1] #band 2 (blue)

        elif self.im_type == 'pan':
            band_1 = self.original_image[self.subimage_index] #band 1
            img_dim = band_1.shape
            subimage = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.uint8)    
            subimage[:,:,0] = self.original_image[self.subimage_index]
            subimage[:,:,1] = self.original_image[self.subimage_index]
            subimage[:,:,2] = self.original_image[self.subimage_index]

        elif self.im_type == 'srgb':
            band_1 = self.original_image[self.subimage_index][:,:,0] #band 1
            img_dim = band_1.shape
            subimage = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.uint8)    
            subimage[:,:,0] = self.original_image[self.subimage_index][:,:,0] #red
            subimage[:,:,1] = self.original_image[self.subimage_index][:,:,1] #green
            subimage[:,:,2] = self.original_image[self.subimage_index][:,:,2] #blue

        ws_subimage = self.secondary_image[self.subimage_index]

        if self.mode == 1:
            current_sp = ws_subimage==self.sp_buffer      #array of 0 or 1 where 1 = current superpixel
            spPosition = np.nonzero(current_sp)           #returns the array position of the superpixel
        if self.mode == 2:
            current_sp = self.sp_buffer
            spPosition = self.sp_buffer

        xMin = np.amin(spPosition[0])-40
        xMax = np.amax(spPosition[0])+40
        yMin = np.amin(spPosition[1])-40
        yMax = np.amax(spPosition[1])+40

        if xMin<0:
            xMin=0
            xMax=xMin+80
        if xMax>=np.shape(subimage)[0]:
            xMax=np.shape(subimage)[0]-1
            xMin=xMax-80
        if yMin<0:
            yMin=0
            yMax=yMin+80
        if yMax>=np.shape(subimage)[1]:
            yMax=np.shape(subimage)[1]-1
            yMin=yMax-80

        #Image 1 (Full zoomed out image)
        full_image = subimage

        #Image 2 (Zoomed in image, no highlighted segment)
        cropped_image = np.copy(subimage)
        cropped_image = cropped_image[xMin:xMax,yMin:yMax] #Cropping

        #Image 3 (Zoomed in image, with segment highlight)
        color_image = np.copy(subimage)
        color_image[:,:,0][current_sp] = 255
        color_image[:,:,2][current_sp] = 0
        color_image = color_image[xMin:xMax,yMin:yMax]

        # Text string to display the user's progress
        progress_counter = "Number Classified: " + str(len(self.label_vector)) + "/" + str(len(self.segment_list))
        
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
        ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelleft='off',
        labelbottom='off')

        ax = self.fig.add_subplot(2,2,2)
        ax.imshow(cropped_image,interpolation='None',vmin=0,vmax=255)
        ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelleft='off',
        labelbottom='off')

        ax = self.fig.add_subplot(2,2,3)
        ax.imshow(full_image,interpolation='None',vmin=0,vmax=255)
        ax.axvspan(yMin,yMax,1.-float(xMax)/np.shape(full_image)[0],1.-float(xMin)/np.shape(full_image)[0],color='red',alpha=0.3)
        ax.set_xlim([0,np.shape(full_image)[1]])
        ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelleft='off',
        labelbottom='off')

        ax = self.fig.add_subplot(2,2,4,adjustable='datalim',frame_on=False)
        # ax.text(0.5,.9,progress_counter,horizontalalignment='center',verticalalignment='center')
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
        # present the user with a random new segment. This is more useful in mode 2, 
        # as this is the primary way of expanding the list. This is not implemented in mode 
        # 1, though it may be useful in the future. 
            
        if self.mode == 1:

            # Add a new set of segments to the list, unless this process has already happened
            # more than x number of times (tracker)
            if len(self.label_vector) == len(self.segment_list):
                if self.tracker >= 5:
                    self.exit_image()
                    return
                self.add_segments()
                self.tracker += 1

            # The current segment is the next one that doesn't have an associated label
            current_segment = self.segment_list[len(self.label_vector)]

            # Exit the training window if we have finished the preset list of segments for this
            # image (only used in mode 1)
            if current_segment[0] != self.image_name:
                print "Finished image %s. Loading next image." %self.image_name
                self.exit_image()
                return

            self.sp_buffer = int(current_segment[2])
            self.subimage_index = int(current_segment[1])

            ws_subimage = self.secondary_image[self.subimage_index]
        

            sp_size = np.sum(ws_subimage==self.sp_buffer)
            
            # Finds the average intensities with a bunch of array comparisons
            if self.im_type == 'wv02_ms': 
                red_intense = np.mean(self.original_image[self.subimage_index][:,:,5][self.sp_buffer==self.secondary_image[self.subimage_index]])
                green_intense = np.mean(self.original_image[self.subimage_index][:,:,3][self.sp_buffer==self.secondary_image[self.subimage_index]])
                blue_intense = np.mean(self.original_image[self.subimage_index][:,:,2][self.sp_buffer==self.secondary_image[self.subimage_index]])
                print "Super Pixel Size: %s" %sp_size
                print "Avg R/G/B Intensity: %i/%i/%i" %(red_intense, green_intense, blue_intense)
            elif self.im_type == 'srgb':
                red_intense = np.mean(self.original_image[self.subimage_index][:,:,0][self.sp_buffer==self.secondary_image[self.subimage_index]])
                green_intense = np.mean(self.original_image[self.subimage_index][:,:,1][self.sp_buffer==self.secondary_image[self.subimage_index]])
                blue_intense = np.mean(self.original_image[self.subimage_index][:,:,2][self.sp_buffer==self.secondary_image[self.subimage_index]])
                print "Super Pixel Size: %s" %sp_size
                print "Avg R/G/B Intensity: %i/%i/%i" %(red_intense, green_intense, blue_intense)
            elif self.im_type == 'pan':
                intense_mean = np.mean(self.original_image[self.subimage_index][self.sp_buffer==self.secondary_image[self.subimage_index]])
                intense_median = np.median(self.original_image[self.subimage_index][self.sp_buffer==self.secondary_image[self.subimage_index]])
                intense_min = np.amin(self.original_image[self.subimage_index][self.sp_buffer==self.secondary_image[self.subimage_index]])
                intense_max = np.amax(self.original_image[self.subimage_index][self.sp_buffer==self.secondary_image[self.subimage_index]])
                print "Super Pixel Size: %s" %sp_size
                print "Mean,Median,Min,Max: %i, %i, %i, %i" %(intense_mean,intense_median,intense_min,intense_max)
        
        if self.mode == 2:
            # Find a new pixel to classify
            while len(self.label_vector) == len(self.segment_list):
                self.add_segments()

            current_segment = self.segment_list[len(self.label_vector)]
            print current_segment

            self.sp_buffer = (current_segment[1],current_segment[2])
            self.subimage_index = int(current_segment[0])


        self.displayImages()

    def previous_super_pixel(self):
        # Make sure this function returns null if there is no previous sp to go back to
        if len(self.label_vector) == 0:
            return
        else:
            # Remove the old classification
            self.label_vector.pop()
            # Update the super pixel index
            current_segment = self.segment_list[len(self.label_vector)]
            if self.mode == 1:
                if current_segment[0] != self.image_name:
                    self.exit_image()
                    return
                self.sp_buffer = int(current_segment[2])
                self.subimage_index = int(current_segment[1])

            elif self.mode == 2:
                self.sp_buffer = (segment_list[len(label_vector)][1],segment_list[len(label_vector)][2])
                self.subimage_index = int(segment_list[len(label_vector)][0])
            # Redraw the display window with the updated indicies. 
            self.displayImages()

    # Add some new segments if we've run out of ones to classify
    def add_segments(self):
        if self.mode == 1:

            j = 0       #counter to prevent too many loops
            segments_to_add = []

            # Loop through subimage until we find one that next 10 candidate segments
            while len(segments_to_add)<10:# and j<200:

                # Reset the counter on every subimage attempt
                segments_to_add = []
                next_subimage = np.random.randint(0,len(self.original_image))
                ws_subimage = self.secondary_image[next_subimage]
                
                # i = np.random.randint(0,np.amax(ws_subimage))
                # len_segments = np.sum(ws_subimage==i)
                # print len_segments
                # segments_to_add.append([self.image_name, next_subimage, i])

                # Cycle through every segment in the current image. If the segment matches
                # our criteria (some dark, some light, some in the middle, and some large)
                # then we add it to the list. Shuffle the indexes so that we select segments
                # from random areas in the images

                dark,light,large,middle = 0,0,0,0       # n for each segment type

                index = range(int(np.amax(ws_subimage)))
                np.random.shuffle(index)
                for i in index:
                    sp_size = np.sum(ws_subimage==i)
                    if sp_size == 0:
                        continue
                    if sp_size > 20:
                        intensity = np.mean(self.original_image[next_subimage][i==ws_subimage])
                        # Skip blank segments
                        if intensity < 1:
                            continue
                        # Add dark segments
                        if intensity < 20 and dark<3:
                            if [self.image_name, next_subimage, i] not in self.segment_list:
                                segments_to_add.append([self.image_name, next_subimage, i])
                                dark += 1
                                continue
                        # Add bright segments
                        if intensity > 200 and light<1:
                            if [self.image_name, next_subimage, i] not in self.segment_list:
                                segments_to_add.append([self.image_name, next_subimage, i])
                                light += 1
                                continue
                        # Add middle intensity segments
                        if intensity >= 20 and intensity < 180 and middle<2:
                            if [self.image_name, next_subimage, i] not in self.segment_list: 
                                segments_to_add.append([self.image_name, next_subimage, i])
                                middle += 1
                                continue
                        # Add large segments
                        if sp_size > 200 and large<2:
                            if [self.image_name, next_subimage, i] not in self.segment_list:
                                segments_to_add.append([self.image_name, next_subimage, i])
                                large += 1

                # Use this if statment for large (WV) images that are primarily
                #   a single surface type. Don't use for smaller aerial images
                # If we didn't get enough points from each category and we've
                # tried less than 5 subimage so far, try another one.
                #if len(segments_to_add)<7 and j<5:
                #    j+=1
                #    continue    # We didn't find enough, so restart
            
                # Fill out the list with random ones if we got enough, or we've tried many subimages.
                #   Use j index to prevent this loop from spending too long looking for points. 
                while (len(segments_to_add)<10 and j<200) and len(segments_to_add)<np.amax(ws_subimage):
                    i = np.random.randint(np.amax(ws_subimage)+1)
                    sp_size = np.sum(ws_subimage==i)
                    if sp_size == 0:
                        j+=1
                        continue
                    intensity = np.mean(self.original_image[next_subimage][i==ws_subimage])
                    # Make sure they're good segments
                    if sp_size > 10 and intensity >= 1:
                        # Prevent adding a duplicate segment
                        if [self.image_name, next_subimage, i] not in self.segment_list:
                            segments_to_add.append([self.image_name, next_subimage, i])
                        else:
                            j+=1
                    else:
                        j+=1

            # Keep only the unique segments
            # print segments_to_add
            self.segment_list += segments_to_add

        # In mode 2 we are doing simple random sampling
        if self.mode == 2:
            found = False
            while found is False:
                self.subimage_index = np.random.randint(0,len(self.original_image))
                # 3rd dimension (b) are the color bands
                if self.im_type == 'pan':
                    [x_size,y_size] = np.shape(self.original_image[self.subimage_index])
                else:
                    [x_size,y_size,b] = np.shape(self.original_image[self.subimage_index])
                x = np.random.randint(0,x_size)
                y = np.random.randint(0,y_size)
                # Pixels that have a value of 0 are empty
                if self.im_type == 'pan':
                    pixel_value = int(self.original_image[self.subimage_index][x,y])
                else:
                    pixel_value = int(self.original_image[self.subimage_index][x,y,0])
                if pixel_value != 0:
                    self.segment_list.append([self.subimage_index,x,y])
                    found = True

    # #def save_subimage():

    
    #Assigning the highlighted superpixel a classification
    def classify(self, keyPress):
        segment_id = int(segment_list[len(label_vector):][0][2])
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
        
        if mode == 1:
            #Creating a feature list for the classified superpixel
            # In development:
            if self.im_type == 'pan':
                entropy_image = entropy(bytescale(self.original_image[self.subimage_index]), disk(4))
                feature_array = feature_calculations.analyze_pan_image(self.original_image[self.subimage_index], 
                    self.secondary_image[self.subimage_index], entropy_image, self.im_date, segment_id=segment_id)
            if self.im_type == 'srgb':
                entropy_image = entropy(bytescale(self.original_image[self.subimage_index][:,:,0]), disk(4))
                feature_array = feature_calculations.analyze_srgb_image(self.original_image[self.subimage_index], 
                    self.secondary_image[self.subimage_index], entropy_image, segment_id=segment_id)
            if self.im_type == 'wv02_ms':
                feature_array = feature_calculations.analyze_ms_image(self.original_image[self.subimage_index], 
                    self.secondary_image[self.subimage_index], segment_id=segment_id)

            # feature_calculations returns a 2d array, but we only want the 1d list of features.
            feature_array = feature_array[0]
            # feature_array = analyze_superpixels(self.original_image[self.subimage_index], self.secondary_image[self.subimage_index], segment_id, self.im_type)
        if mode == 2:
            feature_array = self.secondary_image[self.subimage_index][self.sp_buffer]
            print feature_array, self.label_vector[-1]

        if len(self.feature_matrix) == len(self.label_vector)-1:
            #Adding all of the features found for this watershed to the main matrix
            self.feature_matrix.append(feature_array)
            # print "added a feature"
        else:
            self.feature_matrix[len(self.label_vector)-1] = feature_array
            # print "replaced a feature"

        #Printing some useful statistics
        print str(self.label_vector[-1]) + ": " + keyPress
        print "~"*80

        # print self.original_image[self.subimage_index][self.sp_buffer[0],self.sp_buffer[1],0]
        # print feature_array
        # print "Number with Labels: %s" %len(self.label_vector)
        # print "Number with Features: %s" %len(self.feature_matrix)
        self.next_super_pixel()
    
    #Save progress      
    def save(self):

        print "Saving..."

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
        self.save()
        self.parent.destroy()
        self.parent.quit()

    #Exits the GUI, automatically saves progress
    def quit(self):
        self.save()
        self.parent.destroy()
        self.parent.quit()
        quit()

def create_composite(band_list):

    img_dim = band_list[0].shape
    img = np.zeros((img_dim[0], img_dim[1], len(band_list)), dtype=int)
    for i in range(len(band_list)):
        img[:,:,i] = band_list[i]
    
    return img


# Code from http://chriskiehl.com/article/parallelism-in-one-line/
# Returns a list of .h5 files in the given folder.
def get_image_paths(folder):
    return (os.path.join(folder, f)
        for f in os.listdir(folder)
        if '.h5' in f)

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

def load_data(input_file, mode):
    
    # In mode 1, the data file is the input file
    if mode == 1:
        try:
            data_file = h5py.File(input_file,'r')
        except:
            print "Invalid data file."
            quit()
    # In mode 2, input file is the classified image, and data_file is the 
    # a file in the same folder. 
    if mode == 2:
        data_file_name = os.path.join(input_file[:-14] + '_accuracy_data.h5')
        # If the data_file does not exist in the same folder, create a new
        # one and start the accuracy assessment from scratch
        try:
            data_file = h5py.File(data_file_name,'r')
        except:
            # Don't overwrite a file with the same name
            if os.path.isfile(data_file_name):
                print "Invalid data file."
                quit()
            else:
                print "Existing data file not found: Starting new file."
                feature_matrix = []
                segment_list = []
                data_file = h5py.File(data_file_name,'w')

    # Load the existing feature matrix and segment list if they exist,
    #   otherwise initialize an empty array for these lists.
    if 'feature_matrix' in data_file.keys():
        feature_matrix = data_file['feature_matrix'][:].tolist()
    else:
        feature_matrix = []

    if 'segment_list' in data_file.keys():
        segment_list = data_file['segment_list'][:].tolist()
    else:
        # In mode 1, the segment list is a required entry. Setting this to false
        # triggers the appropriate warning later. 
        if mode == 1:
            segment_list = False
        else:
            segment_list = []

    return data_file, segment_list, feature_matrix

# Allows the user to create a new training set or classification associated with their username
# or to continue building on an existing list. 
def welcome_gui(input_file, segment_type):
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


if __name__ == "__main__":
    
    #### Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="mode 1: folder containing training images | mode 2: classified image file (.h5)")
    parser.add_argument("image_type", type=str, choices=['srgb','wv02_ms','pan'],
                        help="image type: 'srgb', 'wv02_ms', 'pan'")
    parser.add_argument("-m", "--mode", type=int, choices=[1,2], required=True,
                        help="1: create training set | 2: assess accuracy of classified image")
    parser.add_argument("--tds_file", type=str, default=None,
                        help='''Mode 1: Existing training dataset file. Will create a new one with this name if none exists.
                        default: <image_type>_training_data.h5''')
    parser.add_argument("-s", "--splits", type=int, default=9, metavar="int",
                        help='''number of splits to perform on the input images. This is rounded 
                        to the nearest perfect square''')

    #### Parse Arguments
    args = parser.parse_args()
    mode = args.mode
    im_type = args.image_type
    tds_file = args.tds_file
    number_of_splits = args.splits

    if mode == 1:
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

    if mode == 2:
        input_file = args.input

        if not os.path.isfile(input_file):
            print "Could not locate file: %s" %input_file
            quit()

    # input_path is the path of either the training dataset or the accuracy
    input_path = os.path.dirname(input_file)

    
    #### Load Necessary Files
    # Loads the files to use for manual classificaiton. Mode 1 requires the
    # training_validation.h5 file, which contains the list of segments to be
    # classified, and any work already done. Mode 2 requires the accuracy 
    # data file which has any previous work done for that image. 

    # NOTE: In mode 2, segment_list is the ID of manually classified pixels,
    # and feature_matrix is a simple list (not matrix) of the predicted classifications
    data_file, segment_list, feature_matrix = load_data(input_file, mode)

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
        print "Name Error"
        quit()

    data_file.close()


    #### Load the images and segments.

    if mode == 1:
        # Build the list of required images. Finds unique images in the subset of segment list that does not
        # have an associated classification yet. These images are based on the precreated segment list
        # image_list = get_required_images(segment_list[len(label_vector):])

        ## Build a list of the images in the input directory

        # If theres an existing segment list, note the required image to continue the
        # training set
        if segment_list:
            required_images = get_required_images(segment_list[len(label_vector):])
        else:
            required_images = []
        # Add the images in the provided folder to the image list
        image_list = []
        seg_list = list(utils.get_image_paths(input_directory,keyword='segmented.h5'))
        print seg_list
        for ext in utils.valid_extensions:
            raw_list = utils.get_image_paths(input_directory,keyword=ext)
            for raw_im in raw_list:
                if os.path.splitext(raw_im)[0]+"_segmented.h5" not in seg_list:
                    seg_list.append(raw_im)

        image_list = list(set(seg_list))

        print image_list

        # Make sure we have all of the required images
        for image in required_images:
            if image in image_list:
                print "Missing required image: {}".format(image)
                quit()

        # As long as there were files in the input directory, loop indefinitely
        while image_list:

            # Cycle through each image in the list
            for next_image in image_list:

                # If we are out of predetermined segments to classify, start picking
                # images from those in the directory provided
                if len(segment_list) == len(label_vector):
                    image_name = os.path.split(next_image)[1]
                else:
                    image_name = segment_list[len(label_vector)][0]

                image_root, image_ext = os.path.splitext(image_name)
                print image_name
                
                # If the image is already segmented
                if image_name.split('_')[-1] == 'segmented.h5':
                    segmented_name = image_name
                else:
                    # preprocess next_image data
                    image_data, meta_data = prepare_image(input_path, image_name, im_type,
                                    output_path=input_path, verbose=True)
                    # segment image_data
                    print "Segmenting provided image..."
                    segmented_name = os.path.splitext(image_name)[0] + '_segmented.h5'
                    seg_path = os.path.join(input_path,segmented_name)
                    segment_image(image_data, image_type=im_type, write_results=True,
                                    dst_file=seg_path, verbose=True)
                    image_list.append(segmented_name)
                    image_list.remove(next_image)

                h5_file = os.path.join(input_path, segmented_name)
                # from segment import load_from_disk
                original_image = []
                original_image_dict, im_type = load_from_disk(h5_file,False)
                for sub_image in range(len(original_image_dict[1])):
                    # Create a list of image blocks based on the number of bands
                    # in the input image.
                    if im_type == 'wv02_ms':
                        original_image.append(
                            utils.create_composite([original_image_dict[1][sub_image],
                                                    original_image_dict[2][sub_image],
                                                    original_image_dict[3][sub_image],
                                                    original_image_dict[4][sub_image],
                                                    original_image_dict[5][sub_image],
                                                    original_image_dict[6][sub_image],
                                                    original_image_dict[7][sub_image],
                                                    original_image_dict[8][sub_image]])
                                             )
                    if im_type == 'srgb':
                        original_image.append(
                            utils.create_composite([original_image_dict[1][sub_image],
                                                    original_image_dict[2][sub_image],
                                                    original_image_dict[3][sub_image]])
                                             )
                # original_image = utils.create_composite([original_image[1],
                #                                         original_image[2],
                #                                         original_image[3]])

                with h5py.File(h5_file,'r') as f:
                    im_date = f.attrs.get("Image Date")
                    watershed_image = f['watershed'][:]

                print image_name
                # print np.shape(original_image)
                # quit()
                #### Initializing the GUI
                tW = TrainingWindow(original_image, watershed_image, segment_list, 
                    image_name, label_vector, feature_matrix, im_type, im_date, 
                    mode, input_file)

    if mode == 2:

        # image_list = utils.get_image_paths(input_file,keyword='_classified.h5')

        # for image_file in image_list:
        # print "Assessing accuracy of: %s" %image_file

        # Single image of which we are assessing the accuracy
        image_file = h5py.File(input_file, 'r')

        original_image = image_file['original'][:]
        classified_image = image_file['classified'][:]
        im_date = image_file.attrs.get('Image Date')

        image_file.close()

        BLOCK_SIZE = 500

        num_x_subimages = int(original_image.shape[1] / BLOCK_SIZE) #integer truncation intentional
        num_y_subimages = int(original_image.shape[0] / BLOCK_SIZE)
        
        #This will be a list of all the objects created by the subimage class
        original_subimage_list = []
        classified_subimage_list = []

        # These don't include pixels left over after integer truncation, so I changed
        # the splitter function to save all splits in amounts divisible by 500 to get around this for now. 
        for i in range(num_y_subimages):
            for j in range(num_x_subimages):
                original_subimage_list.append(original_image[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE])
                classified_subimage_list.append(classified_image[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE])
            
        original_image = original_subimage_list
        classified_image = classified_subimage_list

        image = None
        savepath = os.path.join(input_file[:-14] + '_accuracy_data.h5')

        #### Initializing the GUI
        tW = TrainingWindow(original_image, classified_image, segment_list, 
                            image, label_vector, feature_matrix, im_type, 
                            im_date, mode, savepath)

        # check_continue = None
        # while check_continue != 'y' or check_continue != 'n':
        #   check_continue = raw_input("Assess another image? (y/n): ")
        #   check_continue = str(check_continue)
        #   if check_continue.lower() == 'n':
        #       quit()



