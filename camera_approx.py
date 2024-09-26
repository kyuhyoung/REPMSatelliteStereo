
import os
import json
from lib.coordinate_system import global_to_local
from lib.rpc_model import RPCModel
from lib.gen_grid import gen_grid
from lib.solve_perspective import solve_perspective
import numpy as np

from lib.latlon_utm_converter import latlon_to_eastnorh


def discretize_volume(rpc_dir, enu_origin_dict):
    with open(rpc_dir) as fp:
        meta = json.load(fp)
    rpc = meta['rpc']
    alt_min = rpc['altOff'] - rpc['altScale']
    alt_max = rpc['altOff'] + rpc['altScale']
    lon_min = rpc['lonOff'] - rpc['lonScale']
    lon_max = rpc['lonOff'] + rpc['lonScale']
    lat_min = rpc['latOff'] - rpc['latScale']
    lat_max = rpc['latOff'] + rpc['latScale']

    # each grid-cell is about 5 meters * 5 meters * 5 meters
    xy_axis_grid_points = 100
    z_axis_grid_points = 50

    # create north_east_height grid
    lat_points = np.linspace(lat_min, lat_max, xy_axis_grid_points)
    lon_points = np.linspace(lon_min, lon_max, xy_axis_grid_points)
    alt_points = np.linspace(alt_min, alt_max, z_axis_grid_points)
    lat_points, lon_points, alt_points = gen_grid(lat_points, lon_points, alt_points)
    xx_utm, yy_utm = latlon_to_eastnorh(lat_points, lon_points)

    xx_enu, yy_enu, zz_enu = global_to_local(lat_points, lon_points, alt_points, enu_origin_dict)

    # convert to local utm
    zz_utm = alt_points
    # convert to enu
    latlonalt = np.hstack((lat_points, lon_points, alt_points))
    utm_local = np.hstack((xx_utm, yy_utm, zz_utm))
    enu = np.hstack((xx_enu, yy_enu, zz_enu))

    return latlonalt, utm_local, enu


class CameraApprox(object):
    def __init__(self, img_dir, rpc_dir, enu_origin_dict, img_name):
        self.img_dir = os.path.join(img_dir, img_name)
        self.rpc_dir = os.path.join(rpc_dir, '{}.json'.format(img_name[:-4]))
        self.img_name = img_name
        # self.latlonalt, self.utm_local = discretize_volume(self.rpc_dir)
        self.latlonalt, self.utm_local, self.enu = discretize_volume(self.rpc_dir, enu_origin_dict)

        with open(os.path.join(rpc_dir, '{}.json'.format(img_name[:-4]))) as fp:
            self.rpc_models = RPCModel(json.load(fp))

    def approx_perspective_enu(self):
        print('deriving a perspective camera approximation...')
        print('scene coordinate frame is in ENU')

        lat_points = self.latlonalt[:, 0:1]
        lon_points = self.latlonalt[:, 1:2]
        alt_points = self.latlonalt[:, 2:3]

        xx = self.enu[:, 0:1]
        yy = self.enu[:, 1:2]
        zz = self.enu[:, 2:3]

        col, row = self.rpc_models.projection(lat_points, lon_points, alt_points)

        # make sure all the points lie inside the image
        width = self.rpc_models.width
        height = self.rpc_models.height
        keep_mask = np.logical_and(col >= 0, row >= 0)
        keep_mask = np.logical_and(keep_mask, col < width)
        keep_mask = np.logical_and(keep_mask, row < height)

        K, R, t = solve_perspective(xx, yy, zz, col, row, keep_mask)

        return K, R, t, xx, yy, zz, col, row, keep_mask, width, height


if __name__ == '__main__':
    pass
