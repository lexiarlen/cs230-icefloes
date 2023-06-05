from fsd_helper import *
import os
import numpy as np
import skimage 
from skimage import io
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import clear_border, expand_labels
from skimage.color import label2rgb
import scipy
from PIL import Image
import pandas as pd

## BEGIN EDITTING
shipside = 'starboard'
date = '20230617'
time = '1235'
## STOP EDITING

# Load image.
fileroot = shipside + '_' + date + '_' + time
orthopath = '/Users/arlenlex/Desktop/' + date + '/' + fileroot + '/' + fileroot + '_ortho/Data/'
print(orthopath)
orthofilepath = os.path.join(orthopath, os.listdir(orthopath)[0])
print(orthofilepath)

# Segment image. 
segmented_im = segmentation(orthofilepath)

# Get FSD output. 
alpha, R2, labeled_im, props = FSD(segmented_im)
fsd_params = {'alpha': alpha, 'R^2': R2}#np.array([alpha, R2])
props_df = pd.DataFrame(props)

# Save data.
outpath = '/Users/arlenlex/Desktop/' + date + '/' + fileroot + '/fsd/'
outpath_seg = outpath + fileroot + '_segmented_im'
outpath_fsd = outpath + fileroot + '_fsd'
outpath_lab = outpath + fileroot + '_labeled_im'
outpath_props = outpath + fileroot + '_props'

np.save(outpath_seg, segmented_im)
np.save(outpath_lab, labeled_im)
np.save(outpath_fsd, fsd_params)
props_df.to_csv(outpath_props)