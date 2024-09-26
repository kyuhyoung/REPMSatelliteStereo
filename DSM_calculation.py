import json
import os
from aggregate_2p5d import run_fuse
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Geometric Correction')
    parser.add_argument('--MVSPath', type=str, default='./DSM_reconstruction/mvs',
                        help='image file')
    parser.add_argument('--enuPath', type=str,
                        default='./DSM_reconstruction/enu_dict.json',
                        help='the origin of enu')
    parser.add_argument('--AOIPath', type=str, default=None,
                        help='Reconstruct the area of interest')
    parser.add_argument('--SavePath', type=str, default='./DSM_reconstruction/dsm',
                        help='save path')
    parser.add_argument('--img_type', type=str,
                        default='3-view',
                        help='3-view, 2-view, multi-data')
    parser.add_argument('--DSM_param', type=list,
                        default=['32','N', '5', '100', '50'],
                        help='["zone_number","hemisphere", "DSM Resolution(meter), "nb_points", "radius_points"]')
    args = parser.parse_args()

    os.makedirs(args.SavePath, exist_ok=True)

    enu_origin = json.load(open(args.enuPath))

    run_fuse(args.MVSPath, args.SavePath, args.DSM_param, enu_origin, imgtype=args.img_type, aoi_path=args.AOIPath)




