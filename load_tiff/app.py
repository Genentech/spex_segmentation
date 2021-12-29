from aicsimageio import AICSImage
from tifffile import TiffFile
import json


def load_tiff(img, is_mibi=True):

    """Load image and check/correct for dimension ordering

    Parameters
    ----------
    image : Path of Multichannel tiff file
    is_mibi:Boolean. Does this image come from MIBI?

    Returns
    -------
    Image Stack : 2D numpy array
    Channels : list

    """
    file = img
    img = AICSImage(img)
    channel_len = max(
        img.size("STCZ")
    )  # It is assumed that the dimension with largest length has the channels
    order = ("S", "T", "C", "Z")
    dim = img.shape

    x = 0
    for x in range(5):
        if dim[x] == channel_len:
            break
    x = x - 1
    order[x]
    string = str(order[x])
    string += "YX"

    if string == "SYX":
        ImageDASK = img.get_image_dask_data(string, T=0, C=0, Z=0)
    if string == "TYX":
        ImageDASK = img.get_image_dask_data(string, S=0, C=0, Z=0)
    if string == "CYX":
        ImageDASK = img.get_image_dask_data(string, S=0, T=0, Z=0)
    if string == "ZYX":
        ImageDASK = img.get_image_dask_data(string, S=0, T=0, C=0)

    ImageTrue = ImageDASK.compute()

    # temporaly
    is_mibi = False
    # temporaly
    if is_mibi == True:
        Channel_list = []
        with TiffFile(file) as tif:
            for page in tif.pages:
                # get tags as json
                description = json.loads(page.tags["ImageDescription"].value)
                Channel_list.append(description["channel.target"])
                # only load supplied channels
                # if channels is not None and description['channel.target'] not in channels:
                # continue

                # read channel data
                # Channel_list.append((description['channel.mass'],description['channel.target']))
    else:
        Channel_list = img.get_channel_names()

    return ImageTrue, Channel_list


def run(**kwargs):

    image = kwargs.get('image_path')
    image, channel = load_tiff(image, is_mibi=True)

    return {
        'median_image': image,
        'channel': channel,
        'image': image
    }
