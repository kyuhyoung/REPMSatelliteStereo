
import os
import json
import numpy as np
from lib.plot_height_map import plot_height_map
from lib.dsm_util import write_dsm_tif
from lib.proj_to_grid import proj_to_grid
import cv2
import open3d as o3d
from osgeo import gdal

def height_fliter( out_dir, DSM_param):
    e_resolution = int(DSM_param[2])  #    meters per pixel
    n_resolution = int(DSM_param[2])

    fliter_path = os.path.join(out_dir, 'dsm_fliter')
    fliter_jpg = os.path.join(out_dir, 'dsm_fliter_jpg')
    os.makedirs(fliter_path, exist_ok=True)
    os.makedirs(fliter_jpg, exist_ok=True)
    dsm_tif_dir = os.path.join(out_dir, 'dsm_tif')
    file_list = os.listdir(dsm_tif_dir)
    for name in file_list:
        dsm_dataset = gdal.Open(os.path.join(dsm_tif_dir, name))
        dsm = dsm_dataset.ReadAsArray()
        geodata = dsm_dataset.GetGeoTransform()
        e_size, n_size  = dsm_dataset.RasterXSize, dsm_dataset.RasterYSize
        ul_e, ul_n = geodata[0],geodata[3]

        # print('ul_e, ul_n:', ul_e, ul_n)
        xx = ul_n - np.arange(n_size) * n_resolution
        yy = ul_e + np.arange(e_size) * e_resolution
        xx, yy = np.meshgrid(xx, yy, indexing='ij')  # xx, yy are of shape (height, width)

        xx = xx.reshape((-1, 1))
        yy = yy.reshape((-1, 1))
        zz = dsm.reshape((-1, 1))
        zz[zz==-10000] = np.nan
    
        valid_mask = np.logical_not(np.isnan(zz)).flatten()
        xx = xx[valid_mask, :]
        yy = yy[valid_mask, :]
        zz = zz[valid_mask, :]
        points = np.concatenate((yy, xx, zz), axis=1)
        # print( points.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        ror_pcd, ind = pcd.remove_radius_outlier(nb_points=int(DSM_param[3]),
                                                 radius=int(DSM_param[4]))
        # ror_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    # std_ratio=5.0)
        points = np.array(ror_pcd.points)
        # print( points.shape)
        assert points.shape[0] != 0
        
        dsm = proj_to_grid(points, ul_e, ul_n, e_resolution, n_resolution, e_size, n_size, propagate=True)
        # median filter
        dsm = cv2.medianBlur(dsm.astype(np.float32), 3)

        fliter_tif = os.path.join(fliter_path, name[:-4] + '.tif')
        write_dsm_tif(dsm, fliter_tif, 
                    (ul_e, ul_n, e_resolution, n_resolution),
                      (DSM_param[0], DSM_param[1]), nodata_val=-10000)
        
        jpg_to_write = os.path.join(fliter_jpg, name[:-4] + '.jpg')
        min_val, max_val = np.nanpercentile(dsm, [1, 99])
        dsm = np.clip(dsm, min_val, max_val)
        plot_height_map(dsm, jpg_to_write, save_cbar=True)


# points is in UTM   tif----aoi
def produce_dsm_from_points( points, DSM_param, tif_to_write, jpg_to_write=None, aoi_dict=None):
    e_resolution = int(DSM_param[2])  # meters per pixel
    n_resolution = int(DSM_param[2])
    if aoi_dict == None:
        ul_e = np.min(points[:, 0])
        ul_n = np.max(points[:, 1])
        lr_e = np.max(points[:, 0])
        lr_n = np.min(points[:, 1])
        e_size = int((lr_e - ul_e) /e_resolution) + 1
        n_size = int((ul_n - lr_n) /n_resolution) + 1
    else:
        ul_e = aoi_dict['ul_easting']
        ul_n = aoi_dict['ul_northing']
        width = aoi_dict['width']
        height = aoi_dict['height']
        re = int(aoi_dict['resolution'])

        lr_e = ul_e + re * width
        lr_n = ul_n - re * height
        e_size = int((lr_e - ul_e) / e_resolution) + 1
        n_size = int((ul_n - lr_n) / n_resolution) + 1

    dsm = proj_to_grid(points, ul_e, ul_n, e_resolution, n_resolution, e_size, n_size, propagate=True)
    # median filter
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)

    write_dsm_tif(dsm, tif_to_write,
                (ul_e, ul_n, e_resolution, n_resolution),
                (DSM_param[0], DSM_param[1]), nodata_val=-10000)

    # create a preview file
    if jpg_to_write is not None:
        min_val, max_val = np.nanpercentile(dsm, [1, 99])
        dsm = np.clip(dsm, min_val, max_val)
        plot_height_map(dsm, jpg_to_write, save_cbar=True)

    return (ul_e, ul_n, e_size, n_size, e_resolution, n_resolution)

# points is in UTM
def produce_dsm_from_height(height, DSM_param, ul_e, ul_n, tif_to_write, jpg_to_write=None):

    e_resolution = int(DSM_param[2])  #    meters per pixel
    n_resolution = int(DSM_param[2])
    n_size, e_size = height.shape[:2]
    
    write_dsm_tif(height, tif_to_write, 
                  (ul_e, ul_n, e_resolution, n_resolution), 
                  (DSM_param[0], DSM_param[1]), nodata_val=-10000)
    # create a preview file
    if jpg_to_write is not None:
        min_val, max_val = np.nanpercentile(height, [1, 99])
        height = np.clip(height, min_val, max_val)
        plot_height_map(height, jpg_to_write, save_cbar=True)

    return ul_e, ul_n, e_size, n_size, e_resolution, n_resolution


if __name__ == '__main__':
    pass
