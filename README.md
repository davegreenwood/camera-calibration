# Calibration Utilities

Python scripts to geometrically calibrate cameras.
Using the algorithms in OpenCV, cameras can be geometrically calibrated
individually, or as a group - such as in a stereo rig. The calibration can be
saved to a file - or to std out. The calibration is valid yaml, so can be used
in other applications using openCV, or elsewhere. The JSON save option includes
the locations of the observed targets in 2d pixel space, and in 3d world space.

The command line interface is invoked with:

    python calibrate.py --help

## Documentation

    Usage: calibrate.py [OPTIONS] COMMAND [ARGS]...

    Options:
    -v, --verbose INTEGER  Level of messaging: 0-3
    -e, --extn TEXT        file extension of images
    -j, --json FILENAME    path to save JSON
    -o, --out FILENAME     Filename to save calibration, or leave empty for
                            stdout. Out is valid yaml - so can be parsed easily.
    --help                 Show this message and exit.

    Commands:
    cam  Calibrate a Camera from a folder that...
    rig  Calibrate a 'Rig' from a list of folders that...

## Calibrate a single camera

A simple example of calibrating a single camera:

    python calibrate.py cam 9 5 0.02  scene_1a/centre

*If there are no corners found in the target images, the most likely reason is
incorrect settings of the target rows or cols.*

    Usage: calibrate.py cam [OPTIONS] TARGET... PATH

    Calibrate a Camera from a folder that contains images of camera
    calibration targets.

    target [int, int, float] -- List of target parameters: rows cols size
    path str -- image directory ...


## Calibrate a rig of many cameras

Calibrating a rig is quite similar, but here pass multiple image folders:

    python calibrate.py rig 9 5 0.02 scene_1a/centre scene_1a/left scene_1a/right

There is an option to perform bundle adjustment on a rig :

    python calibrate.py rig  9 5 0.015 scene_1a/centre scene_1a/left scene_1a/right --bundle


    Usage: calibrate.py rig [OPTIONS] TARGET... [PATHS]...

    Calibrate a 'Rig' from a list of folders that contain images of camera
    calibration targets.

    target [int, int, float] -- List of target parameters: rows cols size
    paths  [str, str, ...] -- List of image directories ...

    Options:
    -bn, --bundle       bundle adjustment of the solution -
                        optimise the 3D positions and intrinsic parameters.
    -cm, --colmap PATH  path to save COLMAP model - a folder with 3 text files.
    --help              Show this message and exit.


## Requirements

The software has a number of dependencies:

* Python > 3.6
* OpenCV > 3.3
* Numpy
* Scipy
* PyYaml
* Click
