import os

import cv2
import torch.utils.data


color_str2cvtr = {
    ("bgr", "rgb"): cv2.COLOR_BGR2RGB,
}


def resize(img, rows, columns):
    # cv2 is (width, height)
    return cv2.resize(img, (columns, rows))


class SegDataset(torch.utils.data.Dataset):
    INPUT_IMG = "input_img"
    TRUTH_IMG = "truth_img"
    INPUT_FNAME = "input_fname"
    TRUTH_FNAME = "truth_fname"

    def __init__(self, input_fldr, truth_fldr=None,
                 img_size=(256, 256),
                 colorspace="RGB",
                 ignore_files={"LICENSE.txt", "ATTRIBUTION.txt"}):
        """Segmentation dataset.

        Assumes that the truth and input folders have same
        order sorted.

        Parameters
        ----------
        img_size : tuple
            Resize to (rows, columns)
        truth_fldr : str, None
            Use None for no truth (testing mode)

        """
        self.has_truth = bool(truth_fldr)  # None, False => False, str => True
        self.color_cvtr = color_str2cvtr[("bgr", colorspace.lower())]
        self.input_fldr = input_fldr
        self.input_files = sorted(
            (f for f in os.listdir(self.input_fldr) if f not in ignore_files))
        self.img_size = img_size
        if self.has_truth:
            self.truth_fldr = truth_fldr
            self.truth_files = sorted(
                (f for f in os.listdir(self.truth_fldr) if f not in ignore_files))
            assert len(self.input_files) == len(self.truth_files), (
                f"Got {len(self.input_files)} input files and "
                f"{len(self.truth_files)} truth files.")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_fname = self.input_files[idx]
        input_path = os.path.join(self.input_fldr, input_fname)
        # cv2 is BGR and we'll probably want HSV, La*b* or at least RGB
        input_img = resize(cv2.imread(input_path), *self.img_size)
        input_img = cv2.cvtColor(input_img, self.color_cvtr, self.color_cvtr)
        sample = {self.INPUT_IMG: input_img,
                  self.INPUT_FNAME: input_fname}
        if self.has_truth:
            truth_fname = self.truth_files[idx]
            truth_path = os.path.join(self.truth_fldr, truth_fname)
            truth_img = resize(cv2.imread(truth_path), *self.img_size)
            sample.update({self.TRUTH_IMG: truth_img,
                           self.TRUTH_FNAME: truth_fname})
        return sample

