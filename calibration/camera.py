import json
import yaml
import numpy as np
import cv2
import logging

from .utils import yield_paths_from_dir, object_points, find_corners_fname
from .utils import rodrigues, CalibUtils, to_numpy, image_size

logger = logging.getLogger(__name__)


def calibrate(img_dir, rows, cols, sqr_size, extn='jpg'):
    params = get_camera_params(img_dir, rows, cols, sqr_size, extn='jpg')
    if not params:
        return
    camera = Camera(params)
    camera.calibrate()
    return camera


def get_camera_params(img_dir, rows, cols, sqr_size, extn='jpg'):
    obj = object_points(rows, cols, sqr_size)
    img_paths, img_pts, obj_pts = [], [], []

    for fname in yield_paths_from_dir(img_dir, extn):
        corner = find_corners_fname(fname, rows, cols)
        if corner is None:
            continue
        img_paths.append(fname)
        img_pts.append(corner)
        obj_pts.append(obj)

    if not img_pts:
        logger.warn('No corners found.')
        return

    logger.info('Found corners in {} images'.format(len(img_pts)))
    return img_pts, obj_pts, img_paths, image_size(fname)


class Camera(CalibUtils):
    """Class that stores the camera data"""
    flags = cv2.CALIB_FIX_ASPECT_RATIO + \
        cv2.CALIB_FIX_PRINCIPAL_POINT + \
        cv2.CALIB_ZERO_TANGENT_DIST + \
        cv2.CALIB_FIX_K3 +\
        cv2.CALIB_FIX_K2

    criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                cv2.TERM_CRITERIA_EPS, 30, 1e-6)

    obj_pts, img_pts, images = [], [], []
    img_size, brd_size, sqr_size = (), (), 1.

    def __init__(self, params=None):
        super(Camera, self).__init__()
        self.K = np.eye(3, dtype=np.float32)
        self.dist = np.zeros(5, dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.t = np.zeros([3, 1])
        self._set()

        if params:
            img_pts, obj_pts, img_paths, img_size = params
            self.img_pts = to_numpy(img_pts)
            self.obj_pts = to_numpy(obj_pts)
            self.images = img_paths
            self.img_size = img_size

    def __repr__(self):
        """return a string representation - actually a yaml string - of
        the camera intrinsic calibration parameters.

        Returns:
            str -- the camera calibration parameters
        """
        d = self.to_dict(compact=True)
        return yaml.dump(d, default_flow_style=False)

    def _set(self):
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.k1 = self.dist[0]
        self.k2 = self.dist[1]

    def world3d(self):
        def f(r, t, p):
            R = rodrigues(r)
            return (R.dot(p.T) + t.reshape(-1, 1)).T
        world_pts = to_numpy(
                    [f(r, t, p) for r, t, p in zip(
                    self.rvecs, self.tvecs, self.obj_pts)])
        self.world_pts = world_pts.reshape([-1, 3])
        return self.world_pts

    def calibrate(self):
        rms, m, d, r, t = cv2.calibrateCamera(self.obj_pts,
                                              self.img_pts,
                                              self.img_size,
                                              np.eye(3),
                                              np.zeros(5),
                                              flags=self.flags,
                                              criteria=self.criteria)
        self.K = to_numpy(m)
        self.dist = to_numpy(d)
        self.rvecs = to_numpy(r)
        self.tvecs = to_numpy(t)
        self.rms = rms
        self._set()
        self.world3d()
        logger.info('Calibration RMS: {}'.format(rms))
        return rms

    def to_dict(self, compact=False):
        """Convert to dictionary without numpy arrays """
        self._set()
        d = dict(fx=float(self.fx), fy=float(self.fy),
                 cx=float(self.cx), cy=float(self.cy),
                 k1=float(self.k1), k2=float(self.k2))
        if compact:
            return d

        d['K'] = self.K.tolist()
        d['dist'] = self.dist.tolist()
        d['img_pts'] = self.img_pts.tolist()
        d['obj_pts'] = self.obj_pts.tolist()
        d['rvecs'] = self.rvecs.tolist()
        d['tvecs'] = self.tvecs.tolist()
        d['images'] = self.images
        d['img_size'] = self.img_size
        d['rms'] = self.rms
        return d

    def colmap_camera(self):
        h, w = self.img_size
        return 2, w, h, np.array([self.fx, self.cx, self.cy, self.k1])
