import argparse
import os

import matplotlib.pyplot as plt

from data import SegDataset
from default_paths import isic_default, task_12_training, task_1_training_gt


def visualize(dset):
    fig, (ax_i, ax_t) = plt.subplots(1, 2)
    print("Press CTRL-C to stop.")
    for sample in dset:
        input_img = sample[dset.INPUT_IMG]
        truth_img = sample[dset.TRUTH_IMG]
        input_fname = sample[dset.INPUT_FNAME]
        truth_fname = sample[dset.TRUTH_FNAME]
        ax_i.imshow(input_img)
        ax_t.imshow(truth_img)
        ax_i.set_title(input_fname)
        ax_t.set_title(truth_fname)
        plt.pause(0.1)
        plt.draw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(isic_default, task_12_training))
    parser.add_argument("--truth", type=str, default=os.path.join(isic_default, task_1_training_gt))
    args = parser.parse_args()
    dset = SegDataset(args.input, args.truth)
    visualize(dset)

