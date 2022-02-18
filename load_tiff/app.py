from aicsimageio import AICSImage
from tifffile import TiffFile
import json


def load_tiff(img, is_mibi=True):

    """Load image and check/correct for dimension ordering

    Parameters
    ----------
    img : Path of Multichannel tiff file
    is_mibi : Boolean. Does this image come from MIBI?

    Returns
    -------
    Image Stack : 2D numpy array
    Channels : list

    """
    file = img
    img = AICSImage(img)

    # It is assumed that the dimension with largest length has the channels
    channel_len = max(img.size("STCZ"))
    order = ["S", "T", "C", "Z"]
    dim = img.shape

    x = 0
    for x in range(len(order)):
        if dim[x] == channel_len:
            break

    orientation = f'{order[x]}YX'

    args = {'T': 0, 'C': 0, 'Z': 0}
    if orientation == "TYX":
        args = {'S': 0, 'C': 0, 'Z': 0}
    if orientation == "CYX":
        args = {'S': 0, 'T': 0, 'Z': 0}
    if orientation == "ZYX":
        args = {'S': 0, 'T': 0, 'C': 0}

    image_true = img.get_image_dask_data(orientation, **args).compute()

    if is_mibi:
        channel_list = []
        with TiffFile(file) as tif:
            for page in tif.pages:
                # get tags as json
                try:
                    description = json.loads(page.tags["ImageDescription"].value)
                    channel_list.append(description["channel.target"])
                except json.decoder.JSONDecodeError:
                    pass
                # only load supplied channels
                # if channels is not None and description['channel.target'] not in channels:
                # continue

                # read channel data
                # Channel_list.append((description['channel.mass'],description['channel.target']))
            if not channel_list:
                channel_list = img.get_channel_names()
    else:
        channel_list = img.get_channel_names()

    return image_true, channel_list


def run(**kwargs):

    image = kwargs.get('image_path')
    image, channel_list = load_tiff(image)

    return {'image': image, 'all_channels': channel_list}
