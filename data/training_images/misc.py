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
    for idx, element in enumerate(np.unique(image)):
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
    print(lookup_path)
    for filename in glob.glob(str(lookup_path)):
        print("> ",filename)
        image = img_as_ubyte(io.imread(filename, as_gray=True))
        if list(np.unique(image)) != [0, 1, 2, 3]:
            image = pixel_to_index(image)
            io.imsave(arr=image, fname=filename, check_contrast=False)

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
    print(lookup_path)
    for filename in glob.glob(str(lookup_path)):
        print("> ",filename)
        image = img_as_ubyte(io.imread(filename, as_gray=True))
        if image is not None:
            io.imsave(arr=image, fname=filename, check_contrast=False)

    return None

if __name__ == "__main__":
    #pixels_to_indexes(folder="labels_rgb", ext="png")
    image_to_grayscale(folder="labels_rgb", ext="png")