
# FastAI and UNet
# - Reference: https://www.youtube.com/watch?v=DKzL4zumFi8

import os
import numpy as np
from fastai.vision.all import *
from datetime import datetime
from pathlib import Path


from skimage.io import imsave, imread
from skimage.transform import rescale
from skimage.util import img_as_bool
from fastai.vision.learner import load_learner

# required by fastai while predicting:
def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return path/"labels"/f"{image.stem}{LABEL_EXT}"

def acc_camvid(inp, targ):
  targ = targ.squeeze(1)
  mask = targ != 0
  return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

def _rescale_image(image_refer, image_to_rescale):
    """Helper function. Rescale image back to original size, according to
    reference."""
    scale_ratio = np.asarray(image_refer.shape[:2]) / np.asarray(image_to_rescale.shape)
    return rescale(image=image_to_rescale, scale=scale_ratio)

def test(image_path, weights):
    print(datetime.now(),'\t Test --> ', image_path)
    image_rgb = imread(image_path)
    learner = load_learner(fname=weights)
    print(datetime.now(),'\t Processing U-net... ', weights)
    _, _, classes = learner.predict(image_rgb)
    print(datetime.now(),'\t Done - Shape', classes.shape)

    # Unpack result segmentation masks    
    back_bin, lepidop_bin, tags_bin, ruler_bin  = np.asarray(classes)[:4]
    back_bin = img_as_bool(_rescale_image(image_rgb, back_bin))
    tags_bin = img_as_bool(_rescale_image(image_rgb, tags_bin))
    ruler_bin = img_as_bool(_rescale_image(image_rgb, ruler_bin))
    lepidop_bin = img_as_bool(_rescale_image(image_rgb, lepidop_bin))
    # Save image
    imsave('{}_bin_image_lepidop.png'.format(os.path.basename(image_path)), lepidop_bin)
    imsave('{}_bin_image_tags.png'.format(os.path.basename(image_path)), tags_bin)
    imsave('{}_bin_image_ruler.png'.format(os.path.basename(image_path)), ruler_bin)
    return


if __name__ == "__main__":

    # Data variables
    cwd = Path.cwd()

    val_image_path = os.path.join(cwd, "battus10", "val_images","images", "IMG_3870.JPG")
    weights_path = os.path.join(cwd, "battus100", "training_images", "battus10_segmentation_test-4classes_resnet34_b4e50s800.pkl")
    test(val_image_path, weights_path)