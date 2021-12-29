import numpy as np
from deepcell.applications import Mesmer


def deepcell_segmentation(image, seg_channels, mpp):

    """Segment image by deepcell deeplearning method

    Parameters
    ----------
    image : Multichannel image as numpy array
    seg_channels: list of indices to use for nuclear segmentation
    mpp: float, micron per pixel

    Returns
    -------
    labels_final : per cell segmentation as numpy array

    """
    temp2 = np.zeros((image.shape[1], image.shape[2]))
    for i in seg_channels:
        temp = image[i]
        temp2 = temp + temp2

    x = temp2
    y = np.expand_dims(x, axis=0)
    pseudoIF = np.stack((y, y), axis=3)

    app = Mesmer()
    y_pred = app.predict(pseudoIF, image_mpp=mpp, compartment="nuclear")

    labels = np.squeeze(y_pred)

    return labels


def run(**kwargs):

    channel_list = kwargs.get('channel_list')
    channel_list.sort()
    median_image = kwargs.get('median_image')
    mpp = float(kwargs.get('mpp'))

    deepcell_label = deepcell_segmentation(median_image, channel_list, mpp)

    return {'labels': deepcell_label}
