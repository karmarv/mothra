
# FastAI and UNet
# - Reference: https://www.youtube.com/watch?v=DKzL4zumFi8

import os
import numpy as np
from fastai.vision.all import *
from fastai2.distributed import *
from pathlib import Path

# Data variables
cwd = Path.cwd()
path = os.path.join(cwd, "training_images")
codes = np.loadtxt(os.path.join(path, "codes.txt"), dtype='str')


def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return os.path.join(path, "labels", f"{image.stem}.png")

# Train function for UNet Segmentation
def train():
    files = get_image_files(os.path.join(path, "images"))
    print("Total Images:", len(files), " \t Sample: ", files[0])
    dls = SegmentationDataLoaders.from_label_func(path, bs=4, fnames=files, label_func=label_func, codes=codes)
    print('Creating Learner, loading the model')
    learner = unet_learner(dls, resnet34).to_fp16()
    callbacks = [
        EarlyStoppingCallback(min_delta=0.001, patience=5)
    ]
    with learner.parallel_ctx(device_ids=[0]):
        # Distribute Train for 15 epochs
        learner.fine_tune(15, freeze_epochs=2, wd=0.01, base_lr=0.0006, cbs=callbacks)
        learner.export('battus10_segmentation_test-4classes_resnet34_b4e15.pkl')
    print('Done')

if __name__ == "__main__":
    train()