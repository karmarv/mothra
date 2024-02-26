import os
import glob
import numpy as np

from pathlib import Path
from skimage import io
from skimage.util import img_as_ubyte


def pixel_to_index(image):
    """Converts pixel values to indexes.
    
    Parameters
    ----------
    image : array-like
        Input image.
    
    Returns
    -------
    image : array-like
        Image with pixels converted to sequential indexes.
    """
    labels_uq = np.unique(image)
    for idx, element in enumerate(labels_uq):
        print(idx, " --> Gray:", element)
        image[image == element] = idx
    return image


def pixels_to_indexes(folder, ext):
    """Converts pixel values to indexes in all images on folder.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder containing images to be converted.
    ext : str
        Extension of input images.

    Returns
    -------
    None
    """
    lookup_path = Path(folder+"/"+f"*.{ext}")
    output_path = lookup_path.parent.parent / "labels"
    print(lookup_path, " --> ", output_path)
    for filename in glob.glob(str(lookup_path)):
        image = img_as_ubyte(io.imread(filename, as_gray=True))
        outfile = output_path / Path(filename).name
        print("> ",filename, " --> ", outfile)
        if list(np.unique(image)) != [0, 1, 2, 3]:
            image = pixel_to_index(image)
            io.imsave(arr=image, fname=outfile, check_contrast=False)

    return None


def image_to_grayscale(folder, ext):
    """Converts images to grayscale for all images in folder.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder containing images to be converted.
    ext : str
        Extension of input images.

    Returns
    -------
    None
    """
    lookup_path = Path(folder+"/"+f"*.{ext}")
    output_path = lookup_path.parent.parent / "labels_gray"
    print(lookup_path, " --> ", output_path)
    for filename in glob.glob(str(lookup_path)):
        image = img_as_ubyte(io.imread(filename, as_gray=True))
        outfile = output_path / Path(filename).name
        print("> ",filename, " --> ", outfile)
        if image is not None:
            print("Gray:", np.unique(image))
            io.imsave(arr=image, fname=outfile, check_contrast=False)

    return None

def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)

def list_leaf_dirs(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      if not dirs:
         print("Leaf: {}".format(root))
   return(paths)

if __name__ == "__main__":
    pixels_to_indexes(folder="battus100/training_images/labels_rgb", ext="png")
    image_to_grayscale(folder="battus100/training_images/labels_rgb", ext="png")
    pixels_to_indexes(folder="battus100/val_images/labels_rgb", ext="png")
    image_to_grayscale(folder="battus100/val_images/labels_rgb", ext="png")
    # list_leaf_dirs(folder = "../UF_museum_data_2023", ext= ".JPG")