import argparse
import multiprocessing as mp
import os
import shutil
from functools import partial

import cv2
import numpy as np

from args import add_input, add_truth
from default_paths import DEFAULT_IGNORE_FILES


SHORT_SIDE_SZ = 300


def load_img(path, greyscale):
    img = cv2.imread(path, not greyscale)
    return img


def resize_img(img, greyscale):
    if greyscale:
        rows, cols = img.shape
    else:
        rows, cols = img.shape[:-1]

    short_side, long_side = np.argsort([rows, cols])

    ratio = SHORT_SIDE_SZ / img.shape[short_side]
    long_side_sz = round(int(img.shape[long_side] * ratio))
    if long_side == 0:
        new_rows, new_columns = long_side_sz, SHORT_SIDE_SZ
    else:
        new_rows, new_columns = SHORT_SIDE_SZ, long_side_sz

    img = cv2.resize(img, (new_columns, new_rows))
    return img


def write_img(path, img):
    cv2.imwrite(path, img)


def process(filename, dir_in, dir_out, greyscale, ignore_files=DEFAULT_IGNORE_FILES):
    if filename not in ignore_files:
        img = load_img(os.path.join(dir_in, filename), greyscale)
        img = resize_img(img, greyscale)
        write_img(os.path.join(dir_out, filename), img)
    else:
        shutil.copyfile(os.path.join(dir_in, filename), os.path.join(dir_out, filename))


def main(dir_in, dir_out, greyscale):
    process_proc = partial(process, dir_in=dir_in, dir_out=dir_out, greyscale=greyscale)
    with mp.Pool() as p:
        p.map(process_proc, os.listdir(dir_in))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_input(parser)
    add_truth(parser)
    add_input(parser, cropped=True, name=("-ci", "--cropped-input"))
    add_truth(parser, cropped=True, name=("-ct", "--cropped-truth"))
    args = parser.parse_args()
    if os.path.isdir(args.cropped_input):
        shutil.rmtree(args.cropped_input)
    os.makedirs(args.cropped_input)

    main(args.input, args.cropped_input, False)
    if os.path.isdir(args.cropped_truth):
        shutil.rmtree(args.cropped_truth)
    os.makedirs(args.cropped_truth)
    main(args.truth, args.cropped_truth, True)
