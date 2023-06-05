## This script contains the functions necessary to get the FSD from an image. 
## The number of bins can be specified by setting num_bins. The default is num_bins = 15.
## The floe size distribution can be computed using effective floe radius or floe area. The default is radius.

import numpy as np
import skimage 
from skimage import io
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import clear_border, expand_labels
from skimage.color import label2rgb
import scipy
from PIL import Image

def segmentation(image_path, num_erosions = 5, diameter_threshold = 800, connectivity = 1, disk_size = 2):
    '''
    input: raw image
    output: segmented image

    This function segments images of ice floes. We make use of skimage.filters.threshold_otsu, 
    skimage.morphology.binary_erosion, and skimage.morphology.diameter_closing. Erosional steps 
    help to distinguish ice floes from on another. The high diameter threshold fills melt ponds. 
    The disk size is a parameter in the erosion and the connectivity is a parameter in the hole 
    filling. For the helicopter photos from HOTRAX 2005, these defaults worked well, but playing 
    with them might result in more accurate segmentations. 
    '''
    # Get image.
    im = io.imread(image_path)

    # Make sure image is grayscale.
    if (np.array(im)).ndim == 3:
        im = skimage.color.rgb2gray(im)

    # Threshold.
    threshold = skimage.filters.threshold_otsu(im)
    simple_threshold = (im < threshold)
    simple_threshold = 1 - simple_threshold

    # Erode.
    i = 0
    eroded_threshold = simple_threshold
    while i < num_erosions:
        eroded_threshold = skimage.morphology.binary_erosion(eroded_threshold, skimage.morphology.disk(disk_size))
        i+=1

    # Fill holes and ponds.
    segmented_im = skimage.morphology.diameter_closing(eroded_threshold, diameter_threshold = diameter_threshold, connectivity = connectivity)
    
    # Return segmentation. 
    return segmented_im

def FSD(segmented_im, num_bins = 15, length_scale = 'radius', clear_borders=True, minimum_floe_area = 5):
    '''
    input: segmented_im
    output: alpha, R^2, labeled image, region properties

    This function return the key parameter alpha of the floe size distribution as well as
    a measure of the power law fit (R^2 value). The user can specify the number of distinct floe sizes
    using the parameter num_bins and the length scale using the parameter length scale. 
    '''
    # Label the segmented image. 
    labeled_im = label(segmented_im, connectivity = 1)
    h, w = labeled_im.shape

    # Clear borders. 
    if clear_borders:
        labeled_im = clear_border(labeled_im)
    
    # Extract region properties from the labeled image.
    props = regionprops(labeled_im)

    # Create an array of floe areas discarding the smallest floes.
    areas = []
    for j in range(0,len(props)):
        areas.append(props[j].area)
    areas = np.sort(np.array(areas))
    idx = np.max(np.argwhere(areas < minimum_floe_area)) 
    r = areas[idx:]

    # Bin floes by appropriate length scale. Do this in a loop to try to hit desired number of bins. 

    if length_scale == 'radius':
        r = np.sqrt(areas * np.pi)
        max_bin_size = np.sqrt(h**2 + w**2) # maximum possible radius by image dimensions
        min_bin_size = 5
    else: 
        max_bin_size = h * w # maximum possible area by image dimensions
        min_bin_size = 5

    length = 0
    nbins = num_bins
    foo = True
    prev_length = 0
    while(length < num_bins and foo):
        bin_sizes = np.logspace(np.log10(min_bin_size), np.log10(max_bin_size), nbins)
        bins = np.zeros(nbins)
        for i in range(0, len(areas)):
            for j in range(1, num_bins):
                if bin_sizes[j-1] < r[i] and r[i] < bin_sizes[j]:
                    bins[j] += 1
                    break
        number_density = bins/bin_sizes
        indices = np.argwhere(number_density != 0)
        length = indices.size
        if length == prev_length:
            foo = False
        prev_length = length
        number_density = number_density[indices].reshape(length,1)
        bin_sizes = bin_sizes[indices].reshape(length)
        nbins += 1

    # Fit power law.
    logx = np.log(bin_sizes)
    logy = np.log(number_density).reshape(logx.size)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(logx, logy)
    alpha = -1 * slope

    return alpha, r_value**2, labeled_im, props