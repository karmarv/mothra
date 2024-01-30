# %% [markdown]
# #### FastAI and UNet
# - Reference: https://www.youtube.com/watch?v=DKzL4zumFi8

import os
import numpy as np
from fastai.vision.all import *
from pathlib import Path

cwd = Path.cwd()
path = os.path.join(cwd.parent, "training_images")

codes = np.loadtxt(os.path.join(path, "codes.txt"), dtype='str')
codes

files = get_image_files(os.path.join(path, "images"))
print("Total Images:", len(files), " \t Sample: ", files[0])

def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return os.path.join(path, "labels", f"{image.stem}.png")

dls = SegmentationDataLoaders.from_label_func(path, bs=4, fnames=files, label_func=label_func, codes=codes)

#dls.show_batch()

# Now, loading the model 
learn = unet_learner(dls, resnet34)

# Train for 6 epochs
learn.fine_tune(15)
learn.show_results(max_n=4, figsize=(7,8))
learn.save('battus10_segmentation_test-4classes_resnet34_b4e15')
