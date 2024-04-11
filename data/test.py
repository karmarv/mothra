
# FastAI and UNet
# - Reference: https://www.youtube.com/watch?v=DKzL4zumFi8

import os
import glob
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

# required by fastai while predicting:
# foreground accuracy measure
def acc_camvid(inp, targ, bkg_idx=0):
  """Computes non-background accuracy for multiclass segmentation"""
  #targ = targ.squeeze(1)
  mask = targ != bkg_idx
  return (inp[mask]>=targ[mask]).mean()

def _rescale_image(image_refer, image_to_rescale):
    """Helper function. Rescale image back to original size, according to
    reference."""
    scale_ratio = np.asarray(image_refer.shape[:2]) / np.asarray(image_to_rescale.shape)
    return rescale(image=image_to_rescale, scale=scale_ratio)

def test(learner, img_file, msk_file=None):
    # Read image file to be tested
    image_rgb = imread(img_file)
    _, _, classes = learner.predict(image_rgb)
    print(datetime.now(),'\t Result shape:', classes.shape)

    # Unpack result segmentation masks    
    back_bin, lepidop_bin, tags_bin, ruler_bin  = np.asarray(classes)[:4]
    back_bin = img_as_bool(_rescale_image(image_rgb, back_bin))
    tags_bin = img_as_bool(_rescale_image(image_rgb, tags_bin))
    ruler_bin = img_as_bool(_rescale_image(image_rgb, ruler_bin))
    lepidop_bin = img_as_bool(_rescale_image(image_rgb, lepidop_bin))

    # Evaluate foreground accuracy metric
    score = 0
    if msk_file is not None:
        mask_ref = imread(msk_file)
        print(datetime.now(),'\t Mask shape:', mask_ref.shape, " Class labels- ", np.unique(mask_ref))
        # Accumulating all foreground classes in prediction results
        pred_bin =  tags_bin + ruler_bin + lepidop_bin
        print(datetime.now(),'\t Pred shape:', pred_bin.shape, ", Foreground labels - ", np.unique(pred_bin))
        score = acc_camvid(mask_ref, pred_bin)

    return  lepidop_bin, tags_bin, ruler_bin, score


if __name__ == "__main__":

    # Data variables
    cwd = Path.cwd()
    weights_path = os.path.join(cwd, "battus100", "training_images", "battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl")
    print(datetime.now(),'\t Loading U-net model from ', weights_path)
    learner = load_learner(fname=weights_path)

    val_image_path = os.path.join(cwd, "battus10", "val_images", "images")
    val_masks_path = os.path.join(cwd, "battus10", "val_images", "labels")
    image_list = glob.glob("{}/*".format(val_image_path))
    for img_file in image_list:
        basename = os.path.splitext(os.path.basename(img_file))
        msk_file = os.path.join(val_masks_path, "{}.png".format(basename[0]))
        print(datetime.now(),'\t Test -> ', img_file, '  Mask -> ', msk_file)
        lepidop_bin, tags_bin, ruler_bin, score = test(learner, img_file, msk_file)
        print(datetime.now(),'\t - Score:', score)
        # Save image for debugging
        #imsave('{}_bin_image_lepidop.png'.format(os.path.basename(image_path)), lepidop_bin)
        #imsave('{}_bin_image_tags.png'.format(os.path.basename(image_path)), tags_bin)
        #imsave('{}_bin_image_ruler.png'.format(os.path.basename(image_path)), ruler_bin)