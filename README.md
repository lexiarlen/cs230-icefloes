# cs230-icefloes
Unet with resnet18 backbone for segmenting sea ice floes. Much of this model was built on https://github.com/usuyama/pytorch-unet.

The trainable parameters of this model are trained on the Global Fiducial sea ice imagery segmented in https://tc.copernicus.org/articles/16/1563/2022/tc-16-1563-2022-discussion.html.

fsd.py can be used to obtain floe size distributions from the segmented imagery.
