from skimage.measure import regionprops_table
import pandas as pd


def feature_extraction(img, labels, channel_list):

    """Extract per cell expression for all channels

    Parameters
    ----------
    img : Multichannel image as numpy array
    labels: 2d segmentation label numpy array
    channel_list: list containing channel names

    Returns
    -------
    perCellDataDF : Pandas dataframe with cell by expression data

    """

    # Image=img
    # img = AICSImage(Image)

    # Get list of channels in ome.tiff
    channels = channel_list
    num_channels = len(channels)

    # Read Image
    # img = load_tiff(image_path)

    # G et coords from labels and create a dataframe to populate mean intensities
    props = regionprops_table(labels, properties=["label", "centroid"])
    per_cell_data_df = pd.DataFrame(props)

    # Loop through each tiff channel and append mean intensity to dataframe
    for x in range(0, num_channels, 1):

        try:
            image = img[x, :, :]

            props = regionprops_table(
                labels, intensity_image=image, properties=["mean_intensity"]
            )
            data_temp = pd.DataFrame(props)
            data_temp.columns = [channels[x]]
            per_cell_data_df = pd.concat([per_cell_data_df, data_temp], axis=1)
        except IndexError:
            print("oops")

    # export and save a .csv file
    # perCellDataDF.to_csv(image+'perCellDataCSV.csv')
    # perCellDataCSV=perCellDataDF.to_csv

    return per_cell_data_df


def run(**kwargs):

    image = kwargs.get('median_image')
    labels = kwargs.get('labels')
    channels_list = kwargs.get('channels_list')

    df = feature_extraction(image, labels, channels_list)

    return {'dataframe': df}
