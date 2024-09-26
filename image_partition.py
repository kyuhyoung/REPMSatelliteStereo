import json
import os
import cv2
import imageio
import numpy as np
from osgeo import gdal

def compute(img, min_percentile, max_percentile):
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness

def uint16_to_uint8(cropped):

    cropped -= cropped.min()
    uint16_img = cropped / (cropped.max() - cropped.min())
    uint16_img *= 255
    uint8_img = uint16_img.astype(np.uint8)
    rgb_img = np.zeros((cropped.shape[0], cropped.shape[1], 3))
    rgb_img[:, :, 0] = uint8_img
    rgb_img[:, :, 1] = uint8_img
    rgb_img[:, :, 2] = uint8_img
    rgb_img = rgb_img.astype('uint8')
    img = aug(rgb_img)
    return img

def rpc_dict(rpc_data):
    rpc_dict = {
        'lonOff': float(rpc_data['LONG_OFF'].split(" ")[0]),
        'lonScale': float(rpc_data['LONG_SCALE'].split(" ")[0]),
        'latOff': float(rpc_data['LAT_OFF'].split(" ")[0]),
        'latScale': float(rpc_data['LAT_SCALE'].split(" ")[0]),
        'altOff': float(rpc_data['HEIGHT_OFF'].split(" ")[0]),
        'altScale': float(rpc_data['HEIGHT_SCALE'].split(" ")[0]),
        'rowOff': float(rpc_data['LINE_OFF'].split(" ")[0]),
        'rowScale': float(rpc_data['LINE_SCALE'].split(" ")[0]),
        'colOff': float(rpc_data['SAMP_OFF'].split(" ")[0]),
        'colScale': float(rpc_data['SAMP_SCALE'].split(" ")[0]),
        'rowNum': np.asarray(rpc_data['LINE_NUM_COEFF'].split(), dtype=np.float64).tolist(),
        'rowDen': np.asarray(rpc_data['LINE_DEN_COEFF'].split(), dtype=np.float64).tolist(),
        'colNum': np.asarray(rpc_data['SAMP_NUM_COEFF'].split(), dtype=np.float64).tolist(),
        'colDen': np.asarray(rpc_data['SAMP_DEN_COEFF'].split(), dtype=np.float64).tolist()
    }
    # meta_dict = {'rpc': rpc_dict,
    #              'height': height,
    #              'width': width,
    #              }
    return rpc_dict

def sub_rpc(start_w, start_h,CropSize, rpc=None):
    if rpc is not None:
        rpc1 = rpc.copy()
        rpc1['colOff'] -= start_w
        rpc1['rowOff'] -= start_h
        subrpc = {
            'rpc': rpc1,
            'height': CropSize,
            'width': CropSize
        }
    return subrpc


def TifCrop(tif_array, SavePath, CropSize, ReRate, name, rpc=None):
    '''
    Sliding window clipping function
    tif_array: remote sensing image
    SavePath: save path
    CropSize: the size of image blooks
    ReRate: RepetitionRate
    rpc: rpc model
    '''
    width = tif_array.shape[1]
    height = tif_array.shape[0]
    img = tif_array

    idx = 0
    os.makedirs(os.path.join(SavePath), exist_ok=True)
    os.makedirs(os.path.join(SavePath, 'images'), exist_ok=True)
    os.makedirs(os.path.join(SavePath, 'metas'), exist_ok=True)

    for i in range(int((height - CropSize * ReRate) / (CropSize * (1 - ReRate)))):
        for j in range(int((width - CropSize * ReRate) / (CropSize * (1 - ReRate)))):
            if (len(img.shape) == 2):
                cropped = img[
                          int(i * CropSize * (1 - ReRate)): int(i * CropSize * (1 - ReRate)) + CropSize,
                          int(j * CropSize * (1 - ReRate)): int(j * CropSize * (1 - ReRate)) + CropSize]
                #  Save image and rpc
                idx = idx + 1
                start_w = j * CropSize * (1 - ReRate)
                start_h = i * CropSize * (1 - ReRate)
                subrpc = sub_rpc(start_w, start_h, CropSize, rpc)

                with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4], idx)), 'w') as fp:
                    json.dump(subrpc, fp, indent=0)
                imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4], idx)), uint16_to_uint8(cropped))

            else:
                cropped = img[
                          int(i * CropSize * (1 - ReRate)): int(i * CropSize * (1 - ReRate)) + CropSize,
                          int(j * CropSize * (1 - ReRate)): int(j * CropSize * (1 - ReRate)) + CropSize,
                          :]
                #  Save image and rpc
                idx = idx + 1
                start_w = j * CropSize * (1 - ReRate)
                start_h = i * CropSize * (1 - ReRate)
                subrpc = sub_rpc(start_w, start_h, CropSize, rpc)
                with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4],idx)), 'w') as fp:
                    json.dump(subrpc, fp, indent=0)
                imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4],idx)), aug(cropped))

    for i in range(int((height - CropSize * ReRate) / (CropSize * (1 - ReRate)))):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - ReRate)): int(i * CropSize * (1 - ReRate)) + CropSize,
                      (width - CropSize): width]
            #  Save image and rpc
            idx = idx + 1
            start_w = (width - CropSize)
            start_h = i * CropSize * (1 - ReRate)
            subrpc = sub_rpc(start_w, start_h, CropSize, rpc)
            with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4], idx)), 'w') as fp:
                json.dump(subrpc, fp, indent=0)
            imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4], idx)), uint16_to_uint8(cropped))

        else:
            cropped = img[
                      int(i * CropSize * (1 - ReRate)): int(i * CropSize * (1 - ReRate)) + CropSize,
                      (width - CropSize): width,
                      :]
            #  Save image and rpc
            idx = idx + 1
            start_w = (width - CropSize)
            start_h = i * CropSize * (1 - ReRate)
            subrpc = sub_rpc(start_w, start_h, CropSize, rpc)
            with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4], idx)), 'w') as fp:
                json.dump(subrpc, fp, indent=0)
            imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4], idx)), aug(cropped))

    for j in range(int((width - CropSize * ReRate) / (CropSize * (1 - ReRate)))):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - ReRate)): int(j * CropSize * (1 - ReRate)) + CropSize]
            #  Save image and rpc
            idx = idx + 1
            start_w = j * CropSize * (1 - ReRate)
            start_h = (height - CropSize)
            subrpc = sub_rpc(start_w, start_h, CropSize, rpc)
            with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4], idx)), 'w') as fp:
                json.dump(subrpc, fp, indent=0)
            imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4], idx)), uint16_to_uint8(cropped))

        else:
            cropped = img[
                      (height - CropSize): height,
                      int(j * CropSize * (1 - ReRate)): int(j * CropSize * (1 - ReRate)) + CropSize,
                      :]

            #  Save image and rpc
            idx = idx + 1
            start_w = j * CropSize * (1 - ReRate)
            start_h = (height - CropSize)
            subrpc = sub_rpc(start_w, start_h, CropSize, rpc)
            with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4],idx)), 'w') as fp:
                json.dump(subrpc, fp, indent=0)
            imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4], idx)), aug(cropped))

    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
        #  Save image and rpc
        idx = idx + 1
        start_w = (width - CropSize)
        start_h = (height - CropSize)
        subrpc = sub_rpc(start_w, start_h, CropSize, rpc)
        with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4], idx)), 'w') as fp:
            json.dump(subrpc, fp, indent=0)
        imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4], idx)), uint16_to_uint8(cropped))

    else:
        cropped = img[
                  (height - CropSize): height,
                  (width - CropSize): width,
                  :]
        #  Save image and rpc
        idx = idx + 1
        start_w = (width - CropSize)
        start_h = (height - CropSize)
        subrpc = sub_rpc(start_w, start_h, CropSize, rpc)
        with open(os.path.join(SavePath, "metas/{}_{}.json".format(name[:-4], idx)), 'w') as fp:
            json.dump(subrpc, fp, indent=0)
        imageio.imwrite(os.path.join(SavePath, 'images/{}_{}.tif'.format(name[:-4], idx)), aug(cropped))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image Partition')
    parser.add_argument('--Path', type=str, default='./image',
                        help='image file and rpc file')
    parser.add_argument('--SavePath', type=str, default='./Partition',
                        help='save path')
    parser.add_argument('--CropSize', type=int, default=2048,
                        help='configuration file')
    parser.add_argument('--RepetitionRate', type=float, default=0.1,
                        help='configuration file')
    args = parser.parse_args()

    files = [x for x in os.listdir(args.Path) if x.endswith('.tif')]
    for name in files:
        print('runing...', os.path.join(args.Path, name))
        dataset = gdal.Open(os.path.join(args.Path, name), gdal.GA_ReadOnly)
        tif_array = dataset.ReadAsArray()
        rpc_data = dataset.GetMetadata("RPC")
        rpc = rpc_dict(rpc_data)
        TifCrop(tif_array, args.SavePath, args.CropSize, args.RepetitionRate, name, rpc)