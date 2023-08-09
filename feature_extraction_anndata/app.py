from skimage.measure import regionprops_table, regionprops
from skimage import measure
import anndata as ad
from anndata import AnnData
from vitessce.data_utils import to_uint8, optimize_adata
import pandas as pd
import numpy as np
import re


def feature_extraction_adata(img, labels,channelnames):
    
    """Extract per cell expression for all channels
    
    Parameters
    ----------
    img : Multichannel image as numpy array
    labels: 2d segmentation label numpy array
    channelnames: list containing channel names (in same order as numpy array!!!)
 
    Returns
    -------
    perCellanndata: anndata single-cell object with all data
    
    """
    
    label_array=labels
    
    props = measure.regionprops_table(label_array,intensity_image=np.transpose(img,(1,2,0)),properties=['label','area','centroid','intensity_mean'])
    perCellData = pd.DataFrame(props)

    perCellData.columns=['cell_id','area_pixels','Y','X']+channelnames  # rename columns

    coordinates=np.array([k for k in perCellData[['X', 'Y']].values.tolist()])

    props = regionprops(label_array)
    ordered_contours=[]
    for region in props:
        # Find contours of the region
        contours = measure.find_contours(label_array == region.label, 0.5)

        if len(contours) > 0:
            # Select the contour with the largest area
            contour = max(contours, key=lambda contour: contour.shape[0])
        else:
            contour=contours[0]

        centroid = np.mean(contour, axis=0)

        # Calculate angles of the points with respect to the centroid
        angles = np.arctan2(contour[:, 0] - centroid[0], contour[:, 1] - centroid[1])

        # Sort points based on angles
        sorted_indices = np.argsort(angles)
        sorted_contour = contour[sorted_indices]

        ordered_contours.append(sorted_contour)

    adata = AnnData(perCellData[channel], obsm={"spatial": coordinates})

    #adata.obs['Image']=[imname] * len(adata.obs)
    adata.obsm['cell_polygon']=np.array(ordered_contours)

    adata.obs['Cell_ID'] = perCellData[['cell_id']].values
    adata.obs['Nucleus_area'] = perCellData[['area_pixels']].values
    adata.obs['x_coordinate']=perCellData[['X']].values
    adata.obs['y_coordinate']=perCellData[['Y']].values


    adata.layers['X_uint8'] = to_uint8(adata.X, norm_along="global")  #vitessce only supports 8bit expression
    
    #adata.write_zarr(args.output)  #Write Anndata file to disk as zarr

    return adata



def run(**kwargs):

    image = kwargs.get('image')
    labels = kwargs.get('labels')
    all_channels = kwargs.get("all_channels", [])


    adata = feature_extraction_adata(image, labels, all_channels)

    return {'dataframe': adata}
