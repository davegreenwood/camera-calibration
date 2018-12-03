import os
import numpy as np
import cv2
import glob
import logging
import json
import yaml


logger = logging.getLogger(__name__)


CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS +
                   cv2.TERM_CRITERIA_MAX_ITER,
                   30, 0.001)


class CalibUtils(object):
    """Some common utils for inheritance"""

    def __init__(self):
        super(CalibUtils, self).__init__()

    def to_dict(self, **kwargs):
        raise NotImplementedError()

    def save_json(self, fname, fid=None):
        """Convert to dictionary of lists, then save to fname."""
        d = self.to_dict()
        if fid:
            json.dump(d, fid)
            return
        with open(fname, 'w') as fid:
            json.dump(d, fid)

    def save_yaml(self, fname, fid=None):
        d = self.to_dict()
        if fid:
            yaml.dump(d, fid, default_flow_style=False)
            return
        with open(fname, 'w') as fid:
            yaml.dump(d, fid, default_flow_style=False)


def yield_paths_from_dir(path, extn='jpg'):
    """Generator of file names from a directory of images."""
    s = '{}/*.{}'.format(path, extn)
    flist = sorted(glob.glob(s))
    for f in flist:
        logger.debug('image path: {}'.format(f))
        yield f


def yield_path_list(path_list, extn='jpg'):
    """From a list of paths, yield a list of one from each sub-directory."""
    for paths in zip(*(yield_paths_from_dir(d) for d in path_list)):
        yield list(paths)


def object_points(rows, cols, square_size):
    points = np.zeros((rows*cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_size
    return to_numpy(points)


def image_size(path):
    """Get the dimensions of an image on disk."""
    img = cv2.imread(path)
    x, y = img.shape[:2]
    return tuple([x, y])


def to_numpy(x):
    """Return x as a numpy array of float32 with no singleton dimensions"""
    return np.asarray(x, dtype=np.float32).squeeze()


def find_corners_fname(fname, rows, cols):
    """Find subpixel chessboard corners in image on disk."""
    image = cv2.imread(fname)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return find_corners(grey, rows, cols)


def find_corners(grey, rows, cols):
    """Find subpixel chessboard corners in greyscale openCV image."""
    ret, corners = cv2.findChessboardCorners(grey, (rows, cols), None)
    if not ret:
        logger.debug('No corners found...')
        return None
    corners = cv2.cornerSubPix(
        grey, corners, (11, 11), (-1, -1),
        criteria=CORNER_CRITERIA)
    return to_numpy(corners)


def target_loss(x, y):
    """loss between two targets - usually lower if matching orientation."""
    loss = np.square((x - x.mean(0)) - (y-y.mean(0))).mean()
    logger.debug('Target loss: {:0.2f}'.format(loss))
    return loss


def align_targets(T):
    """
    Compare the loss for rotated and not rotated targets.
    Return a list of target points rotated such that the loss between
    them is lowest.
    """
    result = [T[0]]
    for t in T[1:]:
        if target_loss(T[0], t) < target_loss(T[0], t[::-1]):
            result.append(t)
        else:
            logger.debug('Target rotated.')
            result.append(t[::-1])
    return result

# -----------------------------------------------------------------------------
# rotations
# -----------------------------------------------------------------------------


def mat2quat(R):
    """ Rotation matrix to quaternion"""
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = R.flat

    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
    ) / 3.0

    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return q if q[0] > 0 else q * -1


def rodrigues(r):
    """Rotation vector to matrix."""
    def S(n):
        Sn = np.array([[0, -n[2], n[1]],
                       [n[2], 0, -n[0]],
                       [-n[1], n[0], 0]])
        return Sn

    theta = np.linalg.norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = S(n)
        R = np.eye(3) + np.sin(theta) * Sn + (1-np.cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = S(r)
        theta2 = theta**2
        R = np.eye(3) + (1-theta2/6.) * Sr + (.5-theta2/24.) * np.dot(Sr, Sr)
    return R


def r2q(r):
    """rotation vector to quaternion"""
    R = rodrigues(r)
    return mat2quat(R)
