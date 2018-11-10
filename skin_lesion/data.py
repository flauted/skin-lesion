import random
import os
from math import floor, ceil
from functools import partial

import cv2
import json
import numpy as np
from PIL import Image
import torch.utils.data
import torchvision.transforms.functional as xF
from torchvision.transforms import ColorJitter

from skin_lesion.default_paths import DEFAULT_IGNORE_FILES


def cvt2sv(image):
    return cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)[:, :, 1:]


def cvt2ab(image):
    return cv2.cvtColor(image, code=cv2.COLOR_BGR2Lab)[:, :, 1:]


color_str2cvtr = {
    ("bgr", "rgb"): partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
    ("bgr", "hsv"): partial(cv2.cvtColor, code=cv2.COLOR_BGR2HSV),
    ("bgr", "sv"): cvt2sv,
    ("bgr", "a*b*"): cvt2ab
}


def resize(img, rows, columns):
    # cv2 is (width, height)
    return cv2.resize(img, (columns, rows))


class SegDataset(torch.utils.data.Dataset):
    IN_CROPPED_IMG = "cropped_img"
    IN_ORIG_IMG = "orig_img"

    GT_CROPPED_IMG = "truth_cropped_img"
    GT_ORIG_IMG = "truth_orig_img"

    IMG_KEYS = [IN_CROPPED_IMG, IN_ORIG_IMG, GT_CROPPED_IMG, GT_ORIG_IMG]

    INPUT_FNAME = "input_fname"
    TRUTH_FNAME = "truth_fname"
    BBOX = "bbox"
    UFM_BBOX = "ufm_bbox"
    RATIOS = "ratios"

    @staticmethod
    def torch_xform(sample):
        for k in [SegDataset.IN_CROPPED_IMG, SegDataset.IN_ORIG_IMG]:
            try:
                sample[k] = xF.to_tensor(sample[k])
            except TypeError:
                pass

        for k in [SegDataset.GT_CROPPED_IMG, SegDataset.GT_ORIG_IMG]:
            try:
                sample[k] = (torch.from_numpy(sample[k]) > 0).to(torch.uint8)
            except TypeError:
                pass
        return sample

    def __init__(self,
                 input_fldr,
                 truth_fldr=None,
                 img_size=(512, 512),
                 colorspace=("RGB", "SV", "a*b*"),
                 jitter=(0.02, 0.02, 0.02, 0.02),
                 ignore_files=DEFAULT_IGNORE_FILES,
                 bbox_file=None,
                 xform=None,
                 random_lr=False,
                 random_ud=False,
                 random_45=True,
                 noised_bbox=True):
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
        self.color_cvtr = []
        for cspace in colorspace:
            self.color_cvtr.append(color_str2cvtr[("bgr", cspace.lower())])
        self.input_fldr = input_fldr
        self.input_files = sorted(
            (f for f in os.listdir(self.input_fldr) if f not in ignore_files))
        self.img_size = img_size
        if self.has_truth:
            self.truth_fldr = truth_fldr
            self.truth_files = [
                os.path.splitext(f)[0] + "_segmentation.png"
                for f in self.input_files
            ]

        self.has_bboxes = False
        if bbox_file is not None:
            with open(bbox_file, "r") as f:
                bbox_data = json.load(f)
            self.has_bboxes = True
            self.bboxes = {}
            for res in bbox_data:
                zero_based_id = res["image_id"] - 1
                isic_id = zero_based_id // 4
                rotation = 90 * (zero_based_id % 4)
                if isic_id not in self.bboxes:
                    self.bboxes[isic_id] = {rotation: []}
                elif rotation not in self.bboxes:
                    self.bboxes[isic_id][rotation] = []

                self.bboxes[isic_id][rotation].append(res)
        else:
            self.has_bboxes = False
            self.bboxes = None

        self.xform = xform
        self.train()
        self.random_lr = random_lr
        self.random_ud = random_ud
        self.random_45 = random_45
        self.noised_bbox = noised_bbox
        if bbox_file is None and self.noised_bbox:
            print("[Warning] No bounding boxes but noised_bbox requires them.")
        self.jitter = ColorJitter(*jitter)

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    @staticmethod
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def _maybe_rotate(sample, keys):
        # images have been rotated by 90s and flipped, so randomly rotating by 45 or not suffices
        if random.choice([True, False]):
            for k in keys:
                try:
                    sample[k] = SegDataset.rotate_image(sample[k], 45)
                except KeyError:
                    pass
        return sample

    @staticmethod
    def _maybe_flip(sample, keys, axis=0):
        if random.choice([True, False]):
            for k in keys:
                try:
                    sample[k] = cv2.flip(sample[k], axis)
                except KeyError:
                    pass
        return sample

    def _jitter(self, input_img):
        return np.array(self.jitter(Image.fromarray(input_img)))

    def maybe_jitter(self, input_img):
        if self._train and self.jitter:
            return self._jitter(input_img)
        else:
            return input_img

    def augment(self, sample):
        if not self.random_lr and not self.random_ud:
            return sample

        if self.random_lr:
            sample = self._maybe_flip(sample, self.IMG_KEYS, axis=1)
        if self.random_ud:
            sample = self._maybe_flip(sample, self.IMG_KEYS, axis=0)
        if self.random_45:
            sample = self._maybe_rotate(sample, self.IMG_KEYS)
        return sample

    def maybe_augment(self, sample):
        if self._train:
            return self.augment(sample)
        else:
            return sample

    def retrieve_bbox(self, input_fname):
        fname = os.path.splitext(input_fname[5:])[0]
        try:
            id_, rot_angle = fname.split("_")
        except ValueError:
            id_, rot_angle = int(fname), 0
        else:
            id_, rot_angle = int(id_), int(rot_angle)
        bboxes = self.bboxes[id_][rot_angle]
        best = max(bboxes, key=lambda x: x['score'])
        bottom, left, width, height = best['bbox']

        bottom, left = floor(bottom), floor(left)
        width, height = ceil(width), ceil(height)

        c0, cf = bottom, bottom + width
        r0, rf = left, left + height
        return {'r0': r0, 'rf': rf, 'c0': c0, 'cf': cf}

    @staticmethod
    def add_percent(bbox, img_size, percent=(0.1, 0.1)):
        r, c = img_size
        r_add = r * percent[0]
        c_add = c * percent[1]
        bbox['c0'] = max(0, int(bbox['c0'] - c_add))
        bbox['cf'] = min(c, int(bbox['cf'] + c_add))
        bbox['r0'] = max(0, int(bbox['r0'] - r_add))
        bbox['rf'] = min(r, int(bbox['rf'] + r_add))

    def calc_ufm_bbox(self, full_bbox, ratios):
        return {k: int(v * ratios[k[0]]) for k, v in full_bbox.items()}

    def crop(self, input_img, best_bbox):
        assert input_img.shape[0] >= best_bbox['rf'] > best_bbox['r0'] >= 0, (
            input_img.shape, best_bbox
        )
        assert input_img.shape[1] >= best_bbox['cf'] > best_bbox['c0'] >= 0, (
            input_img.shape, best_bbox
        )
        input_img = input_img[best_bbox['r0']:best_bbox['rf'],
                    best_bbox['c0']:best_bbox['cf']]
        return input_img

    def __len__(self):
        return len(self.input_files)

    def get_mole(self, idx):
        input_fname = self.input_files[idx]
        input_path = os.path.join(self.input_fldr, input_fname)
        # cv2 is BGR and we'll probably want HSV, La*b* or at least RGB
        raw_img = cv2.imread(input_path)
        if raw_img is None:
            raise FileNotFoundError(input_path)

        raw_img = self.maybe_jitter(raw_img)

        ratios = {'r': self.img_size[0] / raw_img.shape[0],
                  'c': self.img_size[1] / raw_img.shape[1]}

        orig_imgs = []
        for cvtr in self.color_cvtr:
            orig_imgs.append(cvtr(raw_img))
        orig_img = np.concatenate(orig_imgs, axis=-1)

        if self.has_bboxes:
            best_bbox = self.retrieve_bbox(input_fname)
            if self._train and self.noised_bbox:
                grace = (0.1 * random.uniform(0, 2), 0.1 * random.uniform(0, 2))
            else:
                grace = (0.1, 0.1)

            self.add_percent(best_bbox, orig_img.shape[:2], percent=grace)
            ufm_bbox = self.calc_ufm_bbox(best_bbox, ratios)
            cropped_img = self.crop(orig_img, best_bbox)
        else:
            # None or False would make more sense, but this 'hacks' the default batching fn
            # and bool({}) is False.
            best_bbox = {}
            cropped_img = {}
            ufm_bbox = {}

        if cropped_img is not {}:
            ufm_cropped_img = resize(cropped_img, *self.img_size)
        else:
            ufm_cropped_img = {}
        ufm_orig_img = resize(orig_img, *self.img_size)

        sample = {self.IN_CROPPED_IMG: ufm_cropped_img,
                  self.IN_ORIG_IMG: ufm_orig_img,
                  self.BBOX: best_bbox,
                  self.UFM_BBOX: ufm_bbox,
                  self.RATIOS: ratios,
                  self.INPUT_FNAME: input_fname}

        if self.has_truth:
            truth_fname = self.truth_files[idx]
            truth_path = os.path.join(self.truth_fldr, truth_fname)
            orig_gt = cv2.imread(truth_path, 0)
            if orig_gt is None:
                raise FileNotFoundError(truth_path)

            if self.has_bboxes:
                cropped_gt = self.crop(orig_gt, best_bbox)
            else:
                cropped_gt = {}

            if cropped_gt is not {}:
                ufm_cropped_gt = resize(cropped_gt, *self.img_size)
            else:
                ufm_cropped_gt = {}
            ufm_orig_gt = resize(orig_gt, *self.img_size)

            _, ufm_cropped_gt = cv2.threshold(ufm_cropped_gt, 256 // 2, 255, cv2.THRESH_BINARY)
            _, ufm_orig_gt = cv2.threshold(ufm_orig_gt, 256 // 2, 255, cv2.THRESH_BINARY)

            sample.update({self.GT_CROPPED_IMG: ufm_cropped_gt,
                           self.GT_ORIG_IMG: ufm_orig_gt,
                           self.TRUTH_FNAME: truth_fname})
        return sample

    def __getitem__(self, idx):
        sample = self.get_mole(idx)
        sample = self.maybe_augment(sample)

        if self.xform:
            sample = self.xform(sample)

        return sample

