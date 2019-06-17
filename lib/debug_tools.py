from skimage import segmentation, exposure
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from lib import utils


def display_image(raw,watershed,classified,type):

    # Save a color 
    empty_color = [.1,.1,.1]        #Almost black
    snow_color = [.9,.9,.9]         #Almost white
    pond_color = [.31,.431,.647]    #Blue
    gray_color = [.65,.65,.65]          #Gray
    water_color = [0.,0.,0.]        #Black
    shadow_color = [.100, .545, .0]#Orange

    custom_colormap = [empty_color,snow_color,gray_color,pond_color,water_color,shadow_color]
    custom_colormap = colors.ListedColormap(custom_colormap)

    #Making sure there is atleast one of every pixel so the colors map properly (only changes
    # display image, not saved data)
    classified[0][0] = 0
    classified[1][0] = 1
    classified[2][0] = 2
    classified[3][0] = 3
    classified[4][0] = 4
    classified[5][0] = 5

    # Figure that show 3 images: raw, segmented, and classified
    if type == 1:
        # Creating the watershed display image with borders highlighted
        ws_bound = segmentation.find_boundaries(watershed)
        ws_display = utils.create_composite([raw,raw,raw])
        ws_display[:,:,0][ws_bound] = 255
        ws_display[:,:,1][ws_bound] = 255
        ws_display[:,:,2][ws_bound] = 22
    
        fig, axes = plt.subplots(1,3,subplot_kw={'xticks':[], 'yticks':[]})
        fig.subplots_adjust(left=0.05,right=0.99,bottom=0.05,top=0.90,wspace=0.02,hspace=0.2)

        tnrfont = {'fontname':'Times New Roman'}

        axes[0].imshow(raw,cmap='gray',interpolation='None')
        axes[0].set_title("Raw Image", **tnrfont)
        axes[1].imshow(ws_display,interpolation='None')
        axes[1].set_title("Image Segments", **tnrfont)
        axes[2].imshow(classified,cmap=custom_colormap,interpolation='None')
        axes[2].set_title("Classification Output", **tnrfont)

    # Figure that shows 2 images: raw and classified. 
    if type == 2:
        fig, axes = plt.subplots(1,2,subplot_kw={'xticks':[], 'yticks':[]})
        fig.subplots_adjust(hspace=0.3,wspace=0.05)
        axes[0].imshow(raw,interpolation='None')
        axes[0].set_title("Raw Image")
        axes[1].imshow(classified,cmap=custom_colormap,interpolation='None')
        axes[1].set_title("Classification Output")

    plt.show()


# Plots a watershed image on top of and beside the original image
## Used for debugging
def display_watershed(original_data, watershed_data, block=5):

    # block = 5
    watershed = watershed_data[block]
    original_1 = original_data[6][block]
    original_2 = original_data[4][block]
    original_3 = original_data[1][block]

    # randcolor = colors.ListedColormap(np.random.rand(256,3))
    ws_bound = segmentation.find_boundaries(watershed)
    ws_display = utils.create_composite([original_1,original_2,original_3])
    ws_display[:,:,0][ws_bound] = 240
    ws_display[:,:,1][ws_bound] = 80
    ws_display[:,:,2][ws_bound] = 80

    display_im = utils.create_composite([original_1,original_2,original_3])

    fig, axes = plt.subplots(1,2,subplot_kw={'xticks':[], 'yticks':[]})
    fig.subplots_adjust(hspace=0.3,wspace=0.05)

    # axes[1].imshow(self.sobel_image,interpolation='none',cmap='gray')
    axes[0].imshow(display_im,interpolation='none')
    axes[1].imshow(ws_display,interpolation='none')
    plt.show()


def display_histogram(image_band):
    '''
    Displays a histogram of the given band's data. 
    Ignores zero values.
    '''
    hist, bin_centers = exposure.histogram(image_band[image_band>0],nbins=1000)

    plt.figure(1)
    plt.bar(bin_centers, hist)
    # plt.xlim((0,np.max(image_band)))
    # plt.ylim((0,100000))
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


# Method to assess the training set and classification tree used for this classification
def test_training(label_vector, training_feature_matrix):

    print("Size of training set: %i" %len(label_vector))
    print(np.shape(training_feature_matrix))

    # Add a random number to the training data as a reference point
    # Anything less important than a random number is obviously useless
    tfm_new = []
    for i in range(len(training_feature_matrix)):
        tf = training_feature_matrix[i]
        tf.append(np.random.rand(1)[0])
        tfm_new.append(tf)

    training_feature_matrix = tfm_new
    print(np.shape(training_feature_matrix))

    rfc = RandomForestClassifier(n_estimators=100,oob_score=True)
    rfc.fit(training_feature_matrix, label_vector)
    print("OOB Score: %f" %rfc.oob_score_)

    training_feature_matrix = np.array(training_feature_matrix)
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
                axis=0)

    feature_names = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'std7', 'b1/b3', 'b2/b7', 'b4/b7',
                     'ex.b4', 'ex.b8', r'$\frac{b1-b7}{b1+b7}$', r'$\frac{b3-b5}{b3+b5}$',
                     'wb.b1', 'wb.b2', 'wb.b3', 'wb.b4', 'wb.b5', 'wb.b6', 'wb.b7',
                     'bp.b1', 'bp.b2', 'bp.b3', 'bp.b4', 'bp.b5', 'bp.b6', 'bp.b7', 'bp.b8', 'random']

    print(len(feature_names))
    # feature_names = range(len(training_feature_matrix))

    indices = np.argsort(importances)[::-1]

    # feature_names = ['Mean Intensity','Standard Deviation','Size','Entropy','Neighbor Mean Intensity'
    #                   'Neighbor Standard Deviation','Neighbor Maximum Intensity','Neighbor Entropy','Date']

    # Print the feature ranking
    print("Feature ranking:")

    feature_names_sorted = []
    for f in range(training_feature_matrix.shape[1]):
        print("%d. feature %s (%f)" % (f+1, feature_names[indices[f]], importances[indices[f]]))
        feature_names_sorted.append(feature_names[indices[f]])

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(training_feature_matrix.shape[1]), importances[indices],
            color=[.161,.333,.608], yerr=std[indices], align="center", 
            error_kw=dict(ecolor=[.922,.643,.173], lw=2, capsize=3, capthick=2))
    # plt.xticks(range(training_feature_matrix.shape[1]), feature_names_sorted)#, rotation='45')
    plt.xticks(range(training_feature_matrix.shape[1]), feature_names_sorted, rotation='45')
    plt.xlim([-1, training_feature_matrix.shape[1]])
    plt.show()
