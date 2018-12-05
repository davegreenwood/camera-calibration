"""
Module to calibrate camera intrinsics from folder of checkerboard images.
"""
# pylint: disable=E1120
import logging
import click
from calibration.rig import calibrate as calib_rig
from calibration.camera import calibrate as calib_cam

# -----------------------------------------------------------------------------
# module functions
# -----------------------------------------------------------------------------


def set_logger(verbose):
    global logger
    levels = {0: logging.CRITICAL, 1: logging.WARN,
              2: logging.INFO, 3: logging.DEBUG}
    level = levels.get(verbose, logging.DEBUG)
    format = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
    logging.basicConfig(level=level, format=format)
    logger.info('Logger set:')


def parse_target(target):
    rows, cols, sz = target
    return int(rows), int(cols), float(sz)


class Config(object):
    """Default configuration"""

    def __init__(self):
        self.verbose = 3
        self.extn = '.jpg'
        self.json = None
        self.out = '-'


logger = logging.getLogger(__name__)
config = click.make_pass_decorator(Config, ensure=True)

# -----------------------------------------------------------------------------
# CLI functions
# -----------------------------------------------------------------------------


@click.group()
@click.option('--verbose', '-v', default=3,
              help='Level of messaging: 0-3')
@click.option('--extn', '-e', default='.jpg',
              help='file extension of images')
@click.option('--json', '-j', type=click.File('w'), default=None,
              help='path to save JSON')
@click.option('--out', '-o', type=click.File('w'), default='-',
              help='Filename to save calibration, or leave empty for stdout. \
              Out is valid yaml - so can be parsed easily.')
@config
def cli(config, verbose, extn, json, out):
    config.verbose = verbose
    config.extn = extn
    config.json = json
    config.out = out
    set_logger(verbose)


@cli.command()
@click.argument('target', nargs=3)
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
@click.option('--bundle', '-bn', is_flag=True,
                help='bundle adjustment of the solution - \
                optimise the 3D positions and intrinsic parameters.')
@click.option('--colmap', '-cm', type=click.Path(), default=None,
                help='path to save COLMAP model - a folder with 3 text files.')
@config
def rig(config, target, paths, colmap, bundle):
    """
    Calibrate a 'Rig' from a list of folders that contain images of
    camera calibration targets.

    target [int, int, float] -- List of target parameters: rows cols size
    paths  [str, str, ...] -- List of image directories ...
    """
    rows, cols, sz = parse_target(target)
    info = 'Arguments:\n{}\nrows: {}, cols: {}, size:{}'.format(
        paths, rows, cols, sz)
    logger.info(info)
    rig = calib_rig(paths, rows, cols, sz, extn=config.extn)
    if bundle:
        rig.bundle()
    click.echo('#result\n{}'.format(rig), file=config.out)
    if config.json:
        rig.save_json(None, fid=config.json)
    if colmap:
        rig.save_colmap(colmap)


@cli.command()
@click.argument('target', nargs=3)
@click.argument('path', type=click.Path(exists=True), nargs=1)
@config
def cam(config, target, path):
    """
    Calibrate a Camera from a folder that contains images of
    camera calibration targets.

    target [int, int, float] -- List of target parameters: rows cols size
    path  str -- image directory ...
    """

    rows, cols, sz = parse_target(target)
    info = 'images: {}, rows:{}, cols:{}, size:{}'.format(path, rows, cols, sz)
    logger.info(info)
    cam = calib_cam(path, rows, cols, sz, extn=config.extn)
    click.echo('#result\n{}'.format(cam), file=config.out)
    if config.json:
        cam.save_json(None, fid=config.json)
    return

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
