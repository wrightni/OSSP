from skimage import segmentation
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

    # Creating the watershed display image with borders highlighted
    ws_bound = segmentation.find_boundaries(watershed)
    ws_display = utils.create_composite([raw,raw,raw])
    ws_display[:,:,0][ws_bound] = 255
    ws_display[:,:,1][ws_bound] = 255
    ws_display[:,:,2][ws_bound] = 22
    
    # Figure that show 3 images: raw, segmented, and classified
    if type == 1:
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

        axes[0].imshow(raw,cmap='gray',interpolation='None')
        axes[0].set_title("Raw Image")
        axes[1].imshow(classified,cmap=custom_colormap,interpolation='None')
        axes[1].set_title("Classification Output")

    plt.show()


# Plots a watershed image on top of and beside the original image
## Used for debugging
def display_watershed(original_data, watershed_data):

    block = 5
    watershed = watershed_data[block]
    original_1 = original_data[1][block]
    original_2 = original_data[2][block]
    original_3 = original_data[3][block]

    # randcolor = colors.ListedColormap(np.random.rand(256,3))
    ws_bound = segmentation.find_boundaries(watershed)
    ws_display = utils.create_composite([original_1,original_2,original_3])
    ws_display[:,:,0][ws_bound] = 98
    ws_display[:,:,1][ws_bound] = 202
    ws_display[:,:,2][ws_bound] = 202

    display_im = utils.create_composite([original_1,original_2,original_3])

    fig, axes = plt.subplots(1,2,subplot_kw={'xticks':[], 'yticks':[]})
    fig.subplots_adjust(hspace=0.3,wspace=0.05)

    # axes[1].imshow(self.sobel_image,interpolation='none',cmap='gray')
    axes[0].imshow(display_im,interpolation='none')
    axes[1].imshow(ws_display,interpolation='none')
    plt.show()