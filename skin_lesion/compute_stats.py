import argparse

import numpy as np

from args import add_input
from data import SegDataset


def stats(dset):
    sums = np.asarray([0, 0, 0], dtype=np.float32)
    sqsm = np.asarray([0, 0, 0], dtype=np.float32)

    npix = 0
    for elem in dset:
        img = elem["input_img"].astype(np.float32) / 255
        npix += img.size / img.shape[-1]
        sums += img.sum(axis=(0, 1))
        sqsm += np.sum(img ** 2, axis=(0, 1))

    means = sums / npix
    std = np.sqrt(sqsm / npix - means**2)
    print("means:", means)
    print("stds:", std)
    return means, std



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_input(parser, cropped=True)
    args = parser.parse_args()

    dset = SegDataset(args.input, random_lr=False, random_ud=False)
    dset.eval()
    stats(dset)
