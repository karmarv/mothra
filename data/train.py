
# FastAI and UNet
# - Reference: https://www.youtube.com/watch?v=DKzL4zumFi8

import os
# Configuration
num_epochs=20
batch_size=1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import numpy as np
from fastai.vision.all import *
from datetime import datetime
from pathlib import Path

# Data variables
cwd = Path.cwd()
path = os.path.join(cwd, "battus10", "training_images")
codes = np.loadtxt(os.path.join(path, "codes.txt"), dtype='str')
name2id = {v:k for k,v in enumerate(codes)}
print("Unique labels:", name2id)

def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return os.path.join(path, "labels", f"{image.stem}.png")

# Train function for UNet Segmentation
files = get_image_files(os.path.join(path, "images"))
print(datetime.now(), "\t Total Images:", len(files), " \t Sample: ", files[0])
dls = SegmentationDataLoaders.from_label_func(path, bs=batch_size, fnames=files, label_func=label_func, codes=codes, num_workers=0)
print(datetime.now(),'\t Creating Learner, loading the model')

#learner = unet_learner(dls, resnet34)
#learner = unet_learner(dls, resnet18,  pretrained=True)
learner = unet_learner(dls, vgg11_bn,  pretrained=True)

print(datetime.now(),'\t Train model for epochs=', num_epochs)
learner.fine_tune(num_epochs)
print(datetime.now(),'\t Export Model to *.pkl')
# Model export using pickle protocol - https://docs.fast.ai/learner.html#learner
learner.export('battus100_segmentation_test-4classes_vgg11bn_b{}e{}.pkl'.format(batch_size, num_epochs))
print(datetime.now(),'\t Done')
