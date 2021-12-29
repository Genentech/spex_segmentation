from skimage.filters import median
from skimage.feature import peak_local_max
from skimage.morphology import watershed, dilation, erosion, disk
import skimage
from skimage.measure import label
import numpy as np


def classicwatershed_cellseg(img, seg_channels):

    """Detect nuclei in image using classic watershed

    Parameters
    ----------
    img : Multichannel image as numpy array
    seg_channels: list of indices to use for nuclear segmentation

    -------
    Returns
    -------
    dilated_labels : per cell segmentation as numpy array
    """

    temp2 = np.zeros((img.shape[1], img.shape[2]))
    for i in seg_channels:
        temp = img[i]
        temp2 = temp + temp2

    seg_image = temp2 / len(seg_channels)
    med = median(seg_image, disk(3))

    local_max = peak_local_max(med, min_distance=2, indices=False)

    otsu = skimage.filters.threshold_otsu(med)
    otsu_mask = med > otsu

    otsu_mask = skimage.morphology.binary_dilation(otsu_mask, np.ones((2, 2)))
    masked_peaks = local_max * otsu_mask

    seed_label = label(masked_peaks)

    watershed_labels = watershed(
        image=-med,
        markers=seed_label,
        mask=otsu_mask,
        watershed_line=True,
        compactness=20,
    )

    selem = disk(1)
    dilated_labels = erosion(watershed_labels, selem)
    selem = disk(1)
    dilated_labels = dilation(dilated_labels, selem)

    return dilated_labels


def run(**kwargs):

    channel_list = kwargs.get("channel_list", [])
    channel_list.sort()
    median_image = kwargs.get("median_image")

    dilated_labels = classicwatershed_cellseg(median_image, channel_list)

    return {"labels": dilated_labels}
