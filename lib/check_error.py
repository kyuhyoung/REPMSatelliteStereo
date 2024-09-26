
import numpy as np
import correct

def check_perspective_error(xx, yy, zz, col, row, r, q, t, keep_mask,img, rectify=True):
    if keep_mask is not None:
        # logging.info('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    point_cnt = xx.size
    all_ones = np.ones((point_cnt, 1))

    # # check the order of the quantities
    # logging.info('\n')
    # logging.info('fx: {}, fy: {}, cx: {} cy: {}, skew: {}'.format(r[0, 0], r[1, 1], r[0, 2], r[1, 2], r[0, 1]))

    translation = np.tile(t.T, (point_cnt, 1))
    result = np.dot(np.hstack((xx, yy, zz)), q.T) + translation
    cam_xx = result[:, 0:1]
    cam_yy = result[:, 1:2]
    cam_zz = result[:, 2:3]

    # check projection accuracy
    P_hat = np.dot(r, np.hstack((q, t)))
    result = np.dot(np.hstack((xx, yy, zz, all_ones)), P_hat.T)
    esti_col = result[:, 0:1] / result[:, 2:3]
    esti_row = result[:, 1:2] / result[:, 2:3]

    # pixel error
    max_col = np.max(np.abs(esti_col - col))
    max_row = np.max(np.abs(esti_row - row))
    col_err = np.sqrt(np.sum((esti_col - col) ** 2) / len(col))
    row_err = np.sqrt(np.sum((esti_row - row) ** 2) / len(row))
    sigma_err = np.sqrt(col_err ** 2 + row_err ** 2)
    print('RMSE:', sigma_err, 'col_RMSE:',col_err, 'row_RMSE:', row_err)

    src_pts = np.hstack((col, row))
    dst_pts = np.hstack((esti_col, esti_row))

    # 纠正前的误差
    err_col = np.abs(col - esti_col)
    err_row = np.abs(row - esti_row)
    colRMSE_err = np.sqrt(np.sum((esti_col - col) ** 2) / len(col))
    rowRMSE_err = np.sqrt(np.sum((esti_row - row) ** 2) / len(row))
    RMSE = np.sqrt(colRMSE_err ** 2 + rowRMSE_err ** 2)
    project_err = np.sqrt(err_col**2 + err_row**2)

    print('Before correction: proj_err(pixels), min, mean, median, max:{},{},{},{}'.format(np.min(project_err),
                                                                            np.mean(project_err),
                                                                            np.median(project_err), np.max(project_err)))

    if rectify==True:
        err, cor_p, dst, H = correct.correct(img, src_pts, dst_pts, Polynomial=True)  #
        #纠正后的误差
        cor_err_col = err[:, 0]
        cor_err_row = err[:, 1]
        cor_colRMSE = np.sqrt(np.sum(cor_err_col ** 2) / len(col))
        cor_rowRMSE = np.sqrt(np.sum(cor_err_row ** 2) / len(row))
        cor_RMSE = np.sqrt(cor_colRMSE** 2 + cor_rowRMSE ** 2)
        cor_project_err = np.sqrt(cor_err_col**2 + cor_err_row**2)        # L2 范式距离

        print(
            'After correction: proj_err(pixels), min, mean, median, max:{},{},{},{}'.format(np.min(cor_project_err),
                                                                          np.mean(cor_project_err),
                                                                  np.median(cor_project_err), np.max(cor_project_err)))

        erro_rectify = [cor_RMSE, cor_colRMSE,cor_rowRMSE,np.max(cor_project_err), 
                        np.median(cor_project_err),np.mean(cor_project_err)]

    erro = [RMSE, colRMSE_err, rowRMSE_err, np.max(project_err), np.median(project_err),
            np.mean(project_err)]

    # check inverse projection accuracy
    # assume the matching are correct
    result = np.dot(np.hstack((col, row, all_ones)), np.linalg.inv(r.T))
    esti_cam_xx = result[:, 0:1]  # in camera coordinate
    esti_cam_yy = result[:, 1:2]
    esti_cam_zz = result[:, 2:3]

    # compute scale
    scale = (cam_xx * esti_cam_xx + cam_yy * esti_cam_yy + cam_zz * esti_cam_zz) / (
            esti_cam_xx * esti_cam_xx + esti_cam_yy * esti_cam_yy + esti_cam_zz * esti_cam_zz)
    # assert (np.all(scale > 0))
    esti_cam_xx = esti_cam_xx * scale
    esti_cam_yy = esti_cam_yy * scale
    esti_cam_zz = esti_cam_zz * scale

    # inv proj. err
    inv_proj_err = np.sqrt((esti_cam_xx - cam_xx) ** 2 + (esti_cam_yy - cam_yy) ** 2 + (esti_cam_zz - cam_zz) ** 2)
    #print('inv_proj_err (meters), min, mean, median, max: {}, {}'.format(np.min(inv_proj_err), np.mean(inv_proj_err), np.median(inv_proj_err), np.max(inv_proj_err)))

    if rectify==True:
        return np.hstack((erro,erro_rectify)), max_col, max_row,col_err,row_err,sigma_err,H, dst
    else:
        return erro, max_col, max_row,col_err,row_err,sigma_err