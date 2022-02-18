from skimage.measure import regionprops_table
import pandas as pd


def feature_extraction(img, labels, channel_list, all_channels):

    """Extract per cell expression for all channels

    Parameters
    ----------
    img : Multichannel image as numpy array
    labels: 2d segmentation label numpy array
    channel_list: list containing channel index
    all_channels: list of all available channels

    Returns
    -------
    perCellDataDF : Pandas dataframe with cell by expression data

    """

    # Image=img
    # img = AICSImage(Image)

    # G et coords from labels and create a dataframe to populate mean intensities
    props = regionprops_table(labels, properties=["label", "centroid"])
    per_cell_data_df = pd.DataFrame(props)

    # Loop through each tiff channel and append mean intensity to dataframe
    for x in channel_list:

        try:
            image = img[x, :, :]

            props = regionprops_table(
                labels,
                intensity_image=image,
                properties=["mean_intensity"]
            )
            data_temp = pd.DataFrame(props)
            data_temp.columns = [all_channels[x]]
            per_cell_data_df = pd.concat([per_cell_data_df, data_temp], axis=1)
        except IndexError:
            print("oops")

    # export and save a .csv file
    # perCellDataDF.to_csv(image+'perCellDataCSV.csv')
    # perCellDataCSV=perCellDataDF.to_csv

    return per_cell_data_df


def run(**kwargs):

    image = kwargs.get('image')
    labels = kwargs.get('labels')
    channel_list = kwargs.get("channel_list", [])
    all_channels = kwargs.get("all_channels", [])

    if len(channel_list) > 0:
        channel_list: list[int] = [
            all_channels.index(channel)
            for channel in channel_list
            if channel in all_channels
        ]
    else:
        channel_list = list(range(len(all_channels)))

    df = feature_extraction(image, labels, channel_list, all_channels)

    return {'dataframe': df}
