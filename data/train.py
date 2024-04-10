#
# Train a segmentation model
# FastAI and UNet
#
# References:
# - https://www.youtube.com/watch?v=DKzL4zumFi8
# - https://docs.fast.ai/learner.html#learner
#

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"    # export CUDA_VISIBLE_DEVICES=4
import numpy as np
from fastai.vision.all import *
from datetime import datetime
from pathlib import Path


# Configuration
num_epochs  = 50
batch_size  = 8
img_size    = (800, 1200) 

# Train data variables
cwd = Path.cwd()
train_data_path = os.path.join(cwd, "battus100", "training_images")


# Label names 
codes = np.loadtxt(os.path.join(train_data_path, "codes.txt"), dtype='str')
name2id = {v:k for k,v in enumerate(codes)}
background_code = name2id['background']
print(datetime.now(), "\t Unique labels:", name2id)

# Accuracy measure
def acc_camvid(inp, targ):
  targ = targ.squeeze(1)
  mask = targ != background_code
  return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

# Label function
def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return os.path.join(train_data_path, "labels", f"{image.stem}.png")

# Train function for UNet Segmentation
files = get_image_files(os.path.join(train_data_path, "images"))
print(datetime.now(), "\t Loading segmentation masks data for images:", len(files), " \t Sample: ", files[0])
# Dataloader for segmentation masks
dls = SegmentationDataLoaders.from_label_func(train_data_path, bs=batch_size, fnames=files, 
                                            label_func=label_func, 
                                            codes=codes, 
                                            num_workers=4, 
                                            item_tfms=Resize(img_size),
                                            batch_tfms=[*aug_transforms(mult=1.0, 
                                                                        do_flip=True, 
                                                                        flip_vert=True, 
                                                                        max_rotate=90., 
                                                                        min_zoom=1,
                                                                        max_zoom=1.1,
                                                                        max_lighting=0.2,
                                                                        max_warp=0.2, 
                                                                        p_affine=0.75, 
                                                                        p_lighting=0.75, 
                                                                        xtra_tfms=None)])
print(datetime.now(),'\t Creating Learner, initialized the UNET segmentation model')
learner = unet_learner(dls, resnet18,  pretrained=True, metrics=acc_camvid, self_attention=True)
print(datetime.now(),'\t Train model for epochs=', num_epochs)
learner.fine_tune(num_epochs)
print(datetime.now(),'\t Export Model to *.pkl using pickle protocol - https://docs.fast.ai/learner.html#learner')
model_outfile = "battus100_segm_c4_resnet18_b{}_e{}_s{}x{}.pkl".format(batch_size, num_epochs, img_size[1], img_size[0])
learner.export(model_outfile)
print(datetime.now(),'\t Done - ', model_outfile)
