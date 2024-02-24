import matplotlib.pyplot as plt
import numpy as np
import scipy
from osgeo import gdal, ogr, osr
from scipy import io as sio
from sklearn import preprocessing


def normalize(data):
    meanv = np.mean(data, axis=0)
    stdv = np.std(data, axis=0)

    delta = data - meanv
    data = delta / stdv

    return data


def load_datasettest(norm_flag=True):
    imgX = sio.loadmat('river/river_before.mat')['river_before']
    imgY = sio.loadmat('river/river_after.mat')['river_after']

    imgX = np.reshape(imgX, newshape=[-1, imgX.shape[-1]])
    imgY = np.reshape(imgY, newshape=[-1, imgY.shape[-1]])

    GT = sio.loadmat('river/groundtruth.mat')['lakelabel_v1']

    if norm_flag:
        X = preprocessing.StandardScaler().fit_transform(imgX)
        Y = preprocessing.StandardScaler().fit_transform(imgY)
        # X = normalize(imgX)
        # Y = normalize(imgY)

    return X, Y, GT


def load_dataset(path_before='../landsat_test/landsat_0614.tif', path_after='../landsat_test/landsat_0716.tif', train_or_pre='2',
                 norm_flag=True):
    """
    pre-processing the input two images to correct numpy type, two dimentional array according to the bands.
    return values:
    2 dims numpy array X and Y;
    GT:ground true value used to validate the correctness of algorithm;
    row and column: the width and height of image,the width is the vertically direction and
     the height is horizontal direction, both images have the same row and column;
    projection:the projection of two images;
    minx and maxY=y: two diagonal corner of two images
    resolution: space resolution of two images
    """
    suffix = path_before.split('.')[-1]


    if suffix == 'mat':
        imgX = sio.loadmat(path_before)[path_before.split('/')[-1].split('.')[0]]
        imgY = sio.loadmat(path_after)[path_after.split('/')[-1].split('.')[0]]
    elif suffix == 'tif':
        imgX = gdal.Open(path_before)
        imgY = gdal.Open(path_after)
        imgX_proj = imgX.GetProjection()
        minx, xres, xskew, maxy, yskew, yres = imgX.GetGeoTransform()
        img_width = imgX.RasterXSize  # image width
        img_height = imgX.RasterYSize  # image height
        im_bands = imgX.RasterCount
        imgX = imgX.ReadAsArray(0, 0, img_width, img_height)
        imgY = imgY.ReadAsArray(0, 0, img_width, img_height)

        imgX = imgX.transpose(1, 2, 0)
        imgY = imgY.transpose(1, 2, 0)
    else:
        raise Exception('must be .mat or .tif type')

    row, column = imgX.shape[0:2]
    # reshape img based on number of bands
    imgX = np.reshape(imgX, newshape=[-1, imgX.shape[-1]])
    print(imgX.shape)
    imgY = np.reshape(imgY, newshape=[-1, imgY.shape[-1]])

    if norm_flag:
        X = preprocessing.StandardScaler().fit_transform(imgX)
        Y = preprocessing.StandardScaler().fit_transform(imgY)
        # X = normalize(imgX)
        # Y = normalize(imgY)

    # using cva to quickly compute the difference between the two images
    try:
        diff = gdal.Open(r"G:\2308Meteorological Bureau\visiable_Texture_combine\stack\2020GF\stack\diff.tif")
        if train_or_pre == '1':
            minx, xres, xskew, maxy, yskew, yres = diff.GetGeoTransform()
            img_width = diff.RasterXSize  # image width
            img_height = diff.RasterYSize  # image height
            diff = diff.ReadAsArray(0, 0, img_width, img_height)
            diff = diff.reshape(-1)
    except:
        diff=cva(X=X,Y=Y)

    param_dict = {'before': X, 'after': Y,'diff': diff, 'row': row, 'column': column,
                  'projection': imgX_proj, 'minX': minx, 'maxY': maxy, 'resolution': xres}
    return param_dict


def cva(X, Y):
    diff = X - Y
    diff_s = (diff ** 2).sum(axis=-1)
    return np.sqrt(diff_s)


def SFA(X, Y):
    '''
    see http://sigma.whu.edu.cn/data/res/files/SFACode.zip
    '''
    norm_flag = True
    m, n = np.shape(X)
    meanX = np.mean(X, axis=0)
    meanY = np.mean(Y, axis=0)

    stdX = np.std(X, axis=0)
    stdY = np.std(Y, axis=0)

    Xc = (X - meanX) / stdX
    Yc = (Y - meanY) / stdY

    Xc = Xc.T
    Yc = Yc.T

    A = np.matmul((Xc - Yc), (Xc - Yc).T) / m
    B = (np.matmul(Yc, Yc.T) + np.matmul(Yc, Yc.T)) / 2 / m

    D, V = scipy.linalg.eig(A, B)  # V is column wise
    D = D.real
    # idx = D.argsort()
    # D = D[idx]

    if norm_flag is True:
        aux1 = np.matmul(np.matmul(V.T, B), V)
        aux2 = 1 / np.sqrt(np.diag(aux1))
        V = V * aux2
    # V = V[:,0:3]
    X_trans = np.matmul(V.T, Xc).T
    Y_trans = np.matmul(V.T, Yc).T

    return X_trans, Y_trans
