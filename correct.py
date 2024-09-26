
from scipy import linalg
import cv2
import numpy as np

def compute(H, src_pts):

    point_cnt = src_pts.shape[0]
    all_ones = np.ones((point_cnt, 1))
    x = np.float32(src_pts[:,0].reshape(-1,1))
    y = np.float32(src_pts[:,1].reshape(-1,1))
    point = np.hstack((all_ones, x, y, x * y, x * x, y * y))
    dst_pts = np.dot(point, H.T)
    return dst_pts

def svd_factorization(src_pts, dst_pts):
    # SVD
    point_cnt = src_pts.shape[0]
    all_ones = np.ones((point_cnt,1))
    all_zeros = np.zeros((point_cnt, 6))
    x1, y1 = src_pts[:, 0].reshape(-1, 1), src_pts[:, 1].reshape(-1, 1)
    x2, y2 = dst_pts[:, 0].reshape(-1, 1), dst_pts[:, 1].reshape(-1, 1)
    x1, y1 = np.float32(x1), np.float32(y1)
    x2, y2 = np.float32(x2), np.float32(y2)

    B = np.hstack((all_ones, x1, y1, x1 * y1, x1 * x1, y1 * y1))
    T = np.dot(B.T,B)
    R = np.linalg.matrix_rank(T)
    A1 = np.hstack((all_ones, x1, y1, x1 * y1, x1 * x1, y1 * y1,
                    all_zeros))
    A2 = np.hstack((all_zeros,
                   all_ones, x1, y1, x1 * y1, x1 * x1, y1 * y1))
    # A2 = np.hstack((all_zeros,
    #                 all_ones, y1 * y1, x1 * x1, x1 * y1, y1, x1))
    A = np.vstack((A1, A2))
    U, S, V = linalg.svd(A, full_matrices=False)

    b = np.vstack((x2, y2))
    D = np.diag(1 / S)
    H = V.T @ D @ U.T @ b
    H = H.reshape(2,6)

    return H


def correct(img,src_pts,dst_pts, Polynomial =True ):

    width,height = img.shape[1], img.shape[0]

    if Polynomial ==True:
        H = svd_factorization(dst_pts, src_pts)
        correct_tmp = compute(H, dst_pts)
        err = np.abs(src_pts - correct_tmp)

        x, y = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        P = np.hstack((x, y))
        src = compute(H, P)

        col = src[:, 0]
        row = src[:, 1]
        x_src = col.reshape(height, width).astype(np.float32)
        y_src = row.reshape(height, width).astype(np.float32)

        new_img = cv2.remap(img, x_src, y_src, cv2.INTER_LINEAR)
    else:
        point_cnt = src_pts.shape[0]
        all_ones = np.ones((point_cnt,1))
        p1 = src_pts.astype(np.float32)
        p2 = dst_pts.astype(np.float32)
        src_pts = np.float32(p1).reshape(-1, 1, 2)
        dst_pts = np.float32(p2).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 1.0)
        correct_tmp = np.dot(np.hstack((np.float32(p1), all_ones)), H.T)
        correct_col = correct_tmp[:, 0:1] / correct_tmp[:, 2:3]
        correct_row = correct_tmp[:, 1:2] / correct_tmp[:, 2:3]
        cor_p = np.hstack((correct_col, correct_row)).astype(np.float32)
        err = np.abs(p2 - cor_p)
        matchesMask = mask.ravel().tolist()
        cols = img.shape[1]
        rows = img.shape[0]
        new_img = cv2.warpPerspective(img, H, (cols, rows))

    return np.array(err),  correct_tmp, new_img, H       #





