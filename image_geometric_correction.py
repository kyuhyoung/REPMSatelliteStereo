import json
import os
import imageio
from pyquaternion import Quaternion
from camera_approx import CameraApprox
from lib.check_error import check_perspective_error

class ImageCorrect(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def image_geometric(self,K, R, t, xx, yy, zz, col, row, keep_mask, img_name, rectify=True):

        text_ = " "
        img = imageio.v2.imread(os.path.join(self.img_dir, img_name))
        # check approximation error
        tmp = check_perspective_error(xx, yy, zz, col, row, K, R, t, keep_mask, img, rectify)

        if rectify == True:
            err = tmp[0]
            text_ += str(img_name) + "\n"
            text_ += '--------纠正前拟合误差---------' + "\n"
            text_ += 'RMSE: ' + str(err[0]) + "\n"
            text_ += 'RMSE col: ' + str(err[1]) + "\n"
            text_ += 'RMSE row: ' + str(err[2]) + "\n"
            text_ += 'Max: ' + str(err[3]) + "\n"
            text_ += 'Median: ' + str(err[4]) + "\n"
            text_ += 'Mean: ' + str(err[5]) + "\n"
            text_ += '--------------纠正后拟合误差--------------' + "\n"
            text_ += 'RMSE: ' + str(err[6]) + "\n"
            text_ += 'RMSE col: ' + str(err[7]) + "\n"
            text_ += 'RMSE row: ' + str(err[8]) + "\n"
            text_ += 'Max: ' + str(err[9]) + "\n"
            text_ += 'Median: ' + str(err[10]) + "\n"
            text_ += 'Mean: ' + str(err[11]) + "\n"
            text_ += '' + '' + '' + "\n"

            H = tmp[6]
            if H.size == 9:
                H_param = [H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1], H[2, 2]]
                H_dict[img_name] = H_param
            else:
                H_param = [H[0, 0], H[0, 1], H[0, 2], H[0, 3], H[0, 4], H[0, 5],
                           H[1, 0], H[1, 1], H[1, 2], H[1, 3], H[1, 4], H[1, 5]]
                H_dict[img_name] = H_param
            dst = tmp[7]

            return dst, tmp, text_
        else:
            return  tmp

def Get_enu_origin(rpc_dir):
    with open(rpc_dir) as fp:
        meta = json.load(fp)
    rpc = meta['rpc']
    alt_min = rpc['altOff'] - rpc['altScale']
    alt_max = rpc['altOff'] + rpc['altScale']
    lon_min = rpc['lonOff'] - rpc['lonScale']
    lon_max = rpc['lonOff'] + rpc['lonScale']
    lat_min = rpc['latOff'] - rpc['latScale']
    lat_max = rpc['latOff'] + rpc['latScale']

    lat0 = (lat_min + lat_max) / 2.0
    lon0 = (lon_min + lon_max) / 2.0
    alt0 = alt_min
    enu_origin_dict = {'lat_origin': lat0,
                       'lon_origin': lon0,
                       'alt_origin': alt0
                       }
    return enu_origin_dict

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image Geometric Correction')
    parser.add_argument('--ImgPath', type=str, default='..Partition/images',
                        help='image file')
    parser.add_argument('--RPCPath', type=str, default='./Partition/metas',
                        help='rpc file')
    parser.add_argument('--SavePath', type=str, default='./Correct',
                        help='save path')
    args = parser.parse_args()

    files = [x for x in os.listdir(args.ImgPath) if x.endswith('.tif')]

    enu_origin_dict = Get_enu_origin(os.path.join(args.RPCPath, '{}.json'.format(files[0][:-4])))
    perspective_dict, H_dict = {}, {}
    errors_txt = 'img_name, max_col (pixels), max_row (pixels), col_err (pixels), row_err (pixels), ' \
                 'RMSE_err (pixels), Max_err\n'
    text = " "
    for name in files:
        print('runing...', os.path.join(args.ImgPath, name))
        app = CameraApprox(args.ImgPath, args.RPCPath, enu_origin_dict, name)
        K, R, t, xx, yy, zz, col, row,keep_mask, width, height = app.approx_perspective_enu()

        Corapp = ImageCorrect(args.ImgPath)
        dst, tmp, text_ = Corapp.image_geometric(K, R, t, xx, yy, zz, col, row,keep_mask, name, rectify=True)

        # save
        qvec = Quaternion(matrix=R)
        # fx, fy, cx, cy, s, qvec, t
        params = [width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1],
                  qvec[0], qvec[1], qvec[2], qvec[3],
                  t[0, 0], t[1, 0], t[2, 0]]

        name = '{}.tif'.format(name[:-4])
        perspective_dict[name] = params

        os.makedirs(os.path.join(args.SavePath, 'images'), exist_ok=True)
        imageio.imwrite(os.path.join(args.SavePath, 'images', '{}.tif'.format(name[:-4])), dst)

        errors_txt += '{}, {}, {}, {}, {}, {}\n'.format(
            name, tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
        text += text_

    with open(os.path.join(args.SavePath, 'perspective_enu.json'), 'w') as fp:
        json.dump(perspective_dict, fp, indent=2)
    with open(os.path.join(args.SavePath, 'enu_dict.json'), 'w') as fp:
        json.dump(enu_origin_dict, fp, indent=2)
    with open(os.path.join(args.SavePath, 'perspective_enu_error.csv'), 'w') as fp:
        fp.write(errors_txt)
    with open(os.path.join(args.SavePath, 'Correction_parameter.json'), 'w') as fp:
        json.dump(H_dict, fp, indent=2)
    with open(os.path.join(args.SavePath, 'Correction_Error.txt'), 'w') as fp:
        fp.write(text)
