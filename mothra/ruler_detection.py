from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage import color
import numpy as np
from scipy import ndimage as ndi
from joblib import Memory
import matplotlib.patches as patches

from .cache import memory

RULER_TOP = 0.1
RULER_BOT = 0.9
RULER_LEFT = 0.1
RULER_RIGHT = 0.4
FIRST_INDEX_THRESHOLD = 0.9
LINE_WIDTH = 40


def binarize_ruler(ruler_rgb):
    """Returns a binarized version of the image.

    Parameters
    ----------
    ruler_rgb : (M, N) ndarray
        Input image containing the ruler.

    Returns
    -------
    ruler : (M, N) ndarray
        Ruler as a binarized image.

    Notes
    -----
    This performs differently than the U-net; while the U-net returns the
    location of the ruler, this returns the binarized ruler and its elements.
    """
    gray = color.rgb2gray(ruler_rgb)
    thresh = threshold_otsu(gray)
    ruler = gray > thresh

    return ruler


def remove_numbers(focus):
    """Returns a ruler image with the numbers stripped away.

    Parameters
    ----------
    focus : 2D array
        Binary image of the ruler.

    Returns
    -------
    focus_numbers_filled : 2D array
        Binary image of the ruler without numbers.

    Notes
    -----
    The numbers are stripped away to improve the results of the Fourier
    transform, which will process the ruler ticks.
    """
    focus_numbers_markers, _ = ndi.label(focus, ndi.generate_binary_structure(2, 1))
    focus_numbers_regions = regionprops(focus_numbers_markers)
    focus_numbers_region_areas = [region.filled_area for region in focus_numbers_regions]
    focus_numbers_avg_area = np.mean(focus_numbers_region_areas)

    focus_numbers_filled = np.copy(focus)
    for region in focus_numbers_regions:
        if region.eccentricity < 0.99 and region.filled_area > focus_numbers_avg_area:
            min_row, min_col, max_row, max_col = region.bbox
            focus_numbers_filled[min_row:max_row, min_col:max_col] = 0

    return focus_numbers_filled


def fourier(signal, axes=None):
    """Performs a Fourier transform to find the distance in pixels
    between two ticks of the ruler.

    Parameters
    ----------
    signal : 1D array
        Array representing the value of the ticks in space.

    Returns
    -------
    T_space : float
        Distance in pixels between two ticks (0.5 mm).
    """
    # thresholding the signal so the fourier transform results better
    # correlate to frequency, and not amplitude, of the signal
    signal_thresholded = signal > 0

    fourier = np.fft.rfft(signal_thresholded)
    mod = np.abs(fourier)
    mod[0:10] = 0  # we discard the first several coeffs
    freq = np.fft.rfftfreq(len(signal_thresholded))

    f_space = freq[np.argmax(mod)]
    T_space = 1 / f_space

    if axes and axes[4]:
        axes[4].plot(signal, linewidth=0.5)
        axes[5].axvline(x=f_space, color='r', linestyle='dotted', linewidth=1)
        axes[5].plot(freq, mod, linewidth=0.5)

    return T_space


@memory.cache(ignore=['axes'])
def main(image_rgb, ruler_bin, axes=None):
    """Finds the distance between ticks

    Parameters
    ----------
    image_rgb : array
        array representing the image
    ax : array
        array of Axes that show subplots

    Returns
    -------
    t_space : float
        distance between two ticks (.5 mm)
    """
    # preparing figure.
    if axes and axes[0]:
        axes[0].set_title('Final output')
        axes[0].imshow(image_rgb)
        if axes[3]:
            axes[3].set_title('Image structure')
            axes[4].set_title('Ruler signal')
            axes[5].set_title('Fourier transform of ruler signal')
            axes[3].imshow(image_rgb)

    # detecting the top of the ruler.
    ruler_row, ruler_col = np.nonzero(ruler_bin)
    top_ruler = int(ruler_row.min())
    side_ruler = int(ruler_col.min())
    #print("Ruler row(min,max)={},{}, col(min,max)={},{} ".format(ruler_row.min(), ruler_row.max(), ruler_col.min(), ruler_col.max()))

    # returning a binary version of the ruler, numbers and ticks included.
    focus = ~binarize_ruler(image_rgb[ruler_row.min():ruler_row.max(),
                                      ruler_col.min():ruler_col.max()])

    # Removing the numbers in the ruler to denoise the fourier transform analysis
    focus_numbers_filled = remove_numbers(focus)
    #print("Focus shape (y,x)- {}".format(focus_numbers_filled.shape))
    # Cropping the center of the ruler to improve detection
    up_trim = int(RULER_TOP*focus_numbers_filled.shape[0])        # Trim on y-axis (ruler start to end)
    down_trim = int(RULER_BOT*focus_numbers_filled.shape[0])
    left_focus = int(RULER_LEFT*focus_numbers_filled.shape[1])     # Trim on x-axis (ruler gradings towards left)
    right_focus = int(RULER_RIGHT*focus_numbers_filled.shape[1])
    focus_numbers_filled = focus_numbers_filled[up_trim:down_trim, left_focus:right_focus]

    means = np.mean(focus_numbers_filled, axis=1)
    first_index = np.argmax(means > FIRST_INDEX_THRESHOLD * means.max())

    # Fourier transform analysis to give us the pixels between the 1mm ticks
    sums = np.sum(focus_numbers_filled, axis=1)
    # t_space using a scaling factor based on the grading 1mm (1x) or 0.5mm (2x) 
    t_space = 1 * fourier(sums, axes)
    print("T space - {}".format(t_space))
    x_single = [side_ruler + first_index,
                side_ruler + first_index + t_space]
    y = np.array([top_ruler, top_ruler])
    x_mult = [side_ruler + first_index,
              side_ruler + first_index + t_space * 10]
    print("x single - {}".format(x_single))
    print("x multpl - {}".format(x_mult))
    # plotting.
    if axes and axes[0]:
        axes[0].fill_betweenx(x_single, x_single[1], x_single[1] + LINE_WIDTH, color='red', linewidth=0)
        axes[0].fill_betweenx(x_mult, x_mult[1] - LINE_WIDTH, x_mult[1], color='blue', linewidth=0)

    if axes and axes[3]:
        rect = patches.Rectangle((side_ruler, top_ruler+up_trim),
                                 right_focus,
                                 down_trim,
                                 linewidth=1, edgecolor='red', facecolor='none')
        axes[3].axvline(x=side_ruler, color='blue', linestyle='dashed')
        axes[3].add_patch(rect)

    return t_space, side_ruler
