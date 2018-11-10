import argparse
import os

import matplotlib.pyplot as plt

from args import add_valid, add_masks, add_training
from data import SegDataset


def visualize(dset):
    fig, (ax_i, ax_t) = plt.subplots(1, 2)
    print("Press CTRL-C to stop.")
    for sample in dset:
        input_img = sample[dset.INPUT_IMG]
        truth_img = sample[dset.TRUTH_IMG]
        print(truth_img.shape)
        input_fname = sample[dset.INPUT_FNAME]
        truth_fname = sample[dset.TRUTH_FNAME]
        ax_i.imshow(input_img)
        ax_t.imshow(truth_img, cmap="Greys_r")
        ax_i.set_title(input_fname)
        ax_t.set_title(truth_fname)
        plt.pause(0.1)
        plt.draw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--use_train", action="store_true", default=True)
    add_valid(parser)
    add_masks(parser)
    add_training(parser)
    args = parser.parse_args()
    if not args.use_train:
        dset = SegDataset(args.valid, args.truth)
    else:
        dset = SegDataset(args.input, args.truth)
    dset.eval()
    visualize(dset)
