# Calibration Utilities

Python scripts to geometrically calibrate cameras. Using the algorithms in OpenCV, cameras can be geometrically calibrated individually, or as a group - such as in a stereo rig.

The command line interface is invoked with:

    python calibrate.py --help

to produce the documentation:

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

## Requirements

The software has a number of dependencies:

* Python > 3.6
* OpenCV > 3.3
* Numpy
* Click
