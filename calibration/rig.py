import numpy as np
import logging
import os
import json
import yaml
import cv2

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from .utils import yield_path_list, object_points, align_targets
from .utils import find_corners_fname, image_size
from .utils import CalibUtils, to_numpy, mat2quat
from .camera import Camera

logger = logging.getLogger(__name__)


def all_corners(fnames, rows, cols, sqr_size=1.):
    """Find the corners of a camera target visible in a number of images.

    Arguments:
        fnames {list} -- A list of image filenames - taken at the same time -
                                 of the same target.
        rows {int} -- rows in the target
        cols {int} -- columns in the target
        sqr_size {float} -- Size of each target square - any unit.

    Returns:
        None or List -- Returns None if any image has no found corners - OR-
                        returns a list of k sublists, each sublist is :
        [<n*2 2D image points>, <n*3 3D target points>, <filename as a string>]
                        where 'k' is the number of image filenames in fnames.
    """
    corners = [find_corners_fname(p, rows, cols) for p in fnames]
    if any([True for c in corners if c is None]):
        logger.debug('Did NOT find corners in all views.')
        return None
    corners = align_targets(corners)
    obj = object_points(rows, cols, sqr_size)
    logger.debug('Found corners in all views.')
    return [[to_numpy(c), obj, f] for c, f in zip(corners, fnames)]


def calibrate(path_list, rows, cols, sqr_size=1., extn='jpg'):
    results = []
    for paths in yield_path_list(path_list, extn):
        result = all_corners(paths, rows, cols, sqr_size)
        if result is None:
            continue
        results.append(result)
    img_size = image_size(paths[0])
    params = [[*zip(*r)] + [img_size] for r in zip(*results)]
    cameras = [Camera(p) for p in params]
    for c in cameras:
        c.calibrate()
    rig = Rig(cameras)
    rig.calibrate()
    return rig


def stereo_calibrate(A, B, flags=None, criteria=None):
    """Calculate the transform between two cameras"""
    if flags is None:
        flags = cv2.CALIB_FIX_INTRINSIC
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                    cv2.TERM_CRITERIA_EPS, 30, 1e-6)
    [err, _, _, _, _, R, t, E, F] = cv2.stereoCalibrate(
        to_numpy(A.obj_pts),
        to_numpy(A.img_pts),
        to_numpy(B.img_pts),
        to_numpy(A.K),
        to_numpy(A.dist),
        to_numpy(B.K),
        to_numpy(B.dist),
        tuple(A.img_size),
        flags=flags,
        criteria=criteria)
    logger.info('Stereo RMS: {:0.3f}'.format(err))
    d = dict(R=to_numpy(R),
             t=to_numpy(t),
             E=to_numpy(E),
             F=to_numpy(F),
             Q=to_numpy(mat2quat(R)))
    B.R = d['R']
    B.t = d['t']
    return d


class Rig(CalibUtils):
    """A rig is a group of cameras"""

    def __init__(self, cameras):
        super(Rig, self).__init__()
        self.cameras = cameras
        self.transforms = []

    def __repr__(self):
        """Return a string representation - actually a yaml string of
        the camera calibration parameters.

        Returns:
            str -- the camera calibration parameters
        """
        d = self.to_dict(compact=True)
        return yaml.dump(d, default_flow_style=False)

    def calibrate(self):
        """Estimate R, t, E, F between first and all subsequent cameras"""
        if len(self.cameras) < 2:
            logger.warn(
                'Need at least two cameras to do calibration. Quitting...')
            return
        for c in self.cameras[1:]:
            self.transforms.append(stereo_calibrate(self.cameras[0], c))

    def to_dict(self, compact=False):
        """Convert to dictionary without numpy arrays """
        def f(t):
            d = dict()
            for key, value in t.items():
                if compact and key in ['E', 'F', 'Q']:
                    continue
                d[key] = value.squeeze().tolist()
            return d

        cameras = [c.to_dict(compact) for c in self.cameras]
        transforms = [f(t) for t in self.transforms]
        return dict(intrinsics=cameras, extrinsics=transforms)

    def colmap_points3d(self):
        h = '# 3D point list with one line of data per point:\n' + \
            '# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]' + \
            ' as (IMAGE_ID, POINT2D_IDX) \n' + \
            '# Number of points: 0, mean track length: 0\n' + \
            '# < THIS FILE HAS NO DATA >\n'
        return h

    def colmap_images(self):
        """Return a COLMAP images.txt model as a string.

        Returns:
            str -- COLMAP image data
        """
        h = '# Image list with two lines of data per image:\n' + \
            '# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n' + \
            '# POINTS2D[] as (X, Y, POINT3D_ID) < LINE BLANK IN THIS FILE >\n'
        Q = [np.zeros(4)] + [mat2quat(t['R']) for t in self.transforms]
        T = [np.zeros(3)] + [t['t'].ravel() for t in self.transforms]
        r1 = '{} {} {} {} {} {} {} {} {} {}\n\n'
        s = [h]
        for i, (q, t) in enumerate(zip(Q, T)):
            IMAGE_ID, (QW, QX, QY, QZ), (TX, TY, TZ), CAM_ID, NAME = \
                i+1, q, t, i+1, 'image_{}.jpg'.format(i+1)
            r = r1.format(IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAM_ID, NAME)
            s.append(r)
        return ''.join(s)

    def colmap_cameras(self):
        h = '# Camera list with one line of data per camera:\n' + \
            '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n' + \
            '# Number of cameras: {}\n'.format(len(self.cameras))

        def f(c, i):
            y, x = c.img_size
            s = '{} SIMPLE_RADIAL {} {} {} {} {} {}\n'
            return s.format(i, x, y, c.fx, c.cx, c.cy, c.k1)

        s = [h] + [''.join(f(c, i + 1) for i, c in enumerate(self.cameras))]
        return ''.join(s)

    def save_colmap(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, 'cameras.txt'), 'w') as fid:
            fid.write(self.colmap_cameras())
        with open(os.path.join(directory, 'images.txt'), 'w') as fid:
            fid.write(self.colmap_images())
        with open(os.path.join(directory, 'points3D.txt'), 'w') as fid:
            fid.write(self.colmap_points3d())
