import argparse
import math
import os
import random
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

from skin_lesion.args import add_training, add_valid, add_masks, add_model_dir, add_bbox, add_vbbox
import skin_lesion.data as data
from skin_lesion.segnet import SegNet
from skin_lesion.seglab import SegLab
import skin_lesion.kfold as kfold

CHW_TO_HWC = (1, 2, 0)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 4

KFOLD = 10

# these need to become flags
TRAIN_WORKERS = 4
VALID_WORKERS = 4
EPOCHS = 400
BATCH_SIZE = 1
EFFV_BATCH_SIZE = 2
assert EFFV_BATCH_SIZE // BATCH_SIZE == EFFV_BATCH_SIZE / BATCH_SIZE
RANDOM_LR = True
RANDOM_UD = True
OUTPUT_PATH = os.path.join("..", "output")
LOGDIR = os.path.join("..", "tb")


class Grid:
    __slots__ = ["path", "fig", "axes", "n"]

    def __init__(self, path, n_samples):
        self.path = path
        self.fig, self.axes = plt.subplots(n_samples, 7)
        self.n = 0

    def save(self):
        self.fig.savefig(self.path)

    def close(self):
        plt.close(self.fig)

    def push(self, input_orig, input, output, output_uncropped, output_bin, truth, truth_orig):

        # after spending an hour trying to figure out why sample_binry was
        # black, I've decided safety checks are worth the speed penalty.
        assert input_orig.dim() == 3
        assert input.dim() == 3

        input_orig = input_orig.permute(*CHW_TO_HWC)
        input = input.permute(*CHW_TO_HWC)

        assert output.dim() == 2
        assert output_uncropped.dim() == 2
        assert output_bin.dim() == 2

        assert truth.dim() == 2
        assert truth_orig.dim() == 2
        (ino, inp, out, ouu, oub, tru, tro) = self.axes[self.n]
        for ax in self.axes[self.n]:
            ax.set_axis_off()

        plt.axis("off")
        if self.n == 0:
            ino.set_title("i orig")
            inp.set_title("i")
            out.set_title("o")
            ouu.set_title("o unc")
            oub.set_title("b unc")
            tru.set_title("t")
            tro.set_title("t orig")

        ino.imshow(input_orig)
        inp.imshow(input)
        out.imshow(output, cmap="Greys_r")
        ouu.imshow(output_uncropped, cmap="Greys_r")
        oub.imshow(output_bin, cmap="Greys_r")
        tru.imshow(truth, cmap="Greys_r")
        tro.imshow(truth_orig, cmap="Greys_r")

        self.n += 1


def pixel_accuracy(output_bin, truth_bin):
    return (output_bin == truth_bin).sum(dim=(1, 2), dtype=torch.float32) / output_bin[0].numel()


def iou(output_bin, truth_bin):
    i = (output_bin & truth_bin).sum(dim=(1, 2), dtype=torch.float32)
    u = (output_bin | truth_bin).sum(dim=(1, 2))
    iou = i / u.to(torch.float32)
    iou[u == 0] = 0
    return iou


def set_loader_dset(loader, train=False):
    try:
        # subset: no-kfold with joined on-disk train & valid
        if train:
            loader.dataset.dataset.train()
        else:
            loader.dataset.dataset.eval()
    except AttributeError:
        try:
            # concat: kfold with separate on-disk train & valid
            if train:
                loader.dataset.datasets[0].train()
                loader.dataset.datasets[1].train()
            else:
                loader.dataset.datasets[0].eval()
                loader.dataset.datasets[1].eval()
        except AttributeError:
            # regular: no-kfold with separate on-disk train & valid,
            #          kfold with joined on-disk train & valid
            if train:
                loader.dataset.train()
            else:
                loader.dataset.eval()
    return loader


def validate(model, valid_loader, criterion, name, writer=None, step=None):
    model.eval()
    total_xent = 0
    total_acc = 0
    total_iou = 0
    ct = 0
    random_sample = set(random.sample(range(len(valid_loader.dataset)), NUM_SAMPLES))
    grid = Grid(os.path.join(OUTPUT_PATH, name), NUM_SAMPLES)

    valid_loader = set_loader_dset(valid_loader, train=False)

    with torch.no_grad():
        for i, sample in enumerate(valid_loader):
            bbox = sample[data.SegDataset.UFM_BBOX]

            input_orig = sample[data.SegDataset.IN_ORIG_IMG]
            if bbox:
                input = sample[data.SegDataset.IN_CROPPED_IMG]
            else:
                input = input_orig

            ct += input.shape[0]  # batch size
            truth_orig = sample[data.SegDataset.GT_ORIG_IMG].to(DEV)
            if bbox:
                truth = sample[data.SegDataset.GT_CROPPED_IMG].to(DEV)
            else:
                truth = truth_orig
            input = input.to(DEV)

            output = model(input)
            output = output.squeeze(1)

            if bbox:
                output_uncropped = torch.full_like(output, -100)
                for b in range(output_uncropped.shape[0]):
                    rf, r0, cf, c0 = int(bbox['rf'][b]), int(bbox['r0'][b]), int(bbox['cf'][b]), int(bbox['c0'][b])
                    size = (rf - r0, cf - c0)
                    uncropped_b = F.interpolate(output[b].unsqueeze(0).unsqueeze(0), size=size, mode="bilinear")
                    output_uncropped[b, r0:rf, c0:cf] = uncropped_b.squeeze(1)
            else:
                output_uncropped = output

            loss = criterion(output_uncropped, truth_orig)
            total_xent += float(loss)
            output_bin = torch.sigmoid(output_uncropped) >= 0.5
            total_acc += float(pixel_accuracy(output_bin, truth_orig).sum())
            ious = iou(output_bin, truth_orig)
            total_iou += float(ious.sum())

            output = torch.sigmoid(output)
            output_uncropped = torch.sigmoid(output_uncropped)

            for x in range((i * BATCH_SIZE), (i+1) * BATCH_SIZE):
                if x in random_sample:
                    x_mod_bs = x % BATCH_SIZE
                    grid.push(
                        input_orig[x_mod_bs, 0:3],
                        input[x_mod_bs, 0:3],
                        output[x_mod_bs],
                        output_uncropped[x_mod_bs],
                        output_bin[x_mod_bs],
                        truth[x_mod_bs],
                        truth_orig[x_mod_bs])

        if writer is not None:
            assert step is not None
            writer.add_scalar(f"data/{criterion.__name__}", total_xent / ct, step)
            writer.add_scalar("data/accuracy", total_acc / ct, step)
            writer.add_scalar("data/iou", total_iou / ct, step)

        grid.save()
        grid.close()
        print("[{:s}]".format(name),
              "validation: loss = {0:.4f},".format(total_xent / ct),
              "acc = {0:.4f},".format(total_acc / ct),
              "iou = {0:.4f}".format(total_iou / ct))


def dice_loss_w_logits(input, target):
    # https://github.com/pytorch/pytorch/issues/1249
    smooth = 1.

    iflat = torch.sigmoid(input.view(-1))
    tflat = target.view(-1).to(iflat.dtype)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def main():
    parser = argparse.ArgumentParser()
    add_training(parser)
    add_masks(parser)
    add_valid(parser)
    add_model_dir(parser)
    add_bbox(parser)
    add_vbbox(parser)

    parser.add_argument("-c", "--criterion", choices=["xent", "dice"],
                        type=str, default="dice",
                        help="Loss function.")
    args = parser.parse_args()

    dset = data.SegDataset(
        args.input,
        args.truth,
        bbox_file=args.bbox_file,
        xform=data.SegDataset.torch_xform,
        random_ud=RANDOM_UD,
        random_lr=RANDOM_LR)

    if KFOLD == 0:
        if args.valid is None:
            print("Creating random split validation set.")
            train_size = int(round(len(dset) * 0.8))
            valid_size = len(dset) - train_size
            dset_train, dset_valid = random_split(dset, (train_size, valid_size))
        else:
            dset_train = dset
            dset_valid = data.SegDataset(
                args.valid,
                args.truth,
                bbox_file=args.vbbox_file,
                xform=data.SegDataset.torch_xform,
                random_ud=RANDOM_UD,
                random_lr=RANDOM_LR)
        print(f"Training with {len(dset_train)} samples")
        print(f"Validating with {len(dset_valid)} samples")

    else:
        if args.valid is None:
            full_dset = dset
        if args.valid is not None:
            print("[WARNING] Set to use a separate validation dataset and kfold. "
                  "The separate set will be joined with the full set.")
            dset_valid = data.SegDataset(
                args.valid,
                args.truth,
                bbox_file=args.vbbox_file,
                xform=data.SegDataset.torch_xform,
                random_ud=RANDOM_UD,
                random_lr=RANDOM_LR)
            full_dset = torch.utils.data.ConcatDataset([dset, dset_valid])
        print(f"Training & k-fold x-valing with {len(full_dset)} samples and {KFOLD} folds")


    model = SegLab(1, in_channels=7, pretrained=True, model_dir=args.model_dir)
    print(model)
    model = model.to(DEV)
    if DEV.type == "cuda":
        print("Using GPU")
    model.train()

    if KFOLD == 0:
        train_loader = DataLoader(
            dset_train, batch_size=BATCH_SIZE, num_workers=TRAIN_WORKERS,
            shuffle=True, pin_memory=False)
        valid_loader = DataLoader(
            dset_valid, batch_size=BATCH_SIZE, num_workers=VALID_WORKERS,
            pin_memory=False)
    else:
        sampler = kfold.KfoldSampler(len(full_dset), KFOLD)
        sampler.train = True
        train_loader = DataLoader(
            full_dset, batch_size=BATCH_SIZE, num_workers=TRAIN_WORKERS,
            sampler=sampler, pin_memory=False)
        valid_loader = DataLoader(
            full_dset, batch_size=BATCH_SIZE, num_workers=VALID_WORKERS,
            sampler=sampler, pin_memory=False)

    if args.criterion == "xent":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.criterion == "dice":
        criterion = dice_loss_w_logits
    else:
        raise NotImplementedError(f"Criterion '{args.criterion}' not implemented.")
    optim = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.92)

    UPDATE_EVERY = 1000

    valid_writer = SummaryWriter(log_dir=os.path.join(LOGDIR, "valid"))
    train_writer = SummaryWriter(log_dir=os.path.join(LOGDIR, "train"))

    batches_per_epoch = math.ceil(len(train_loader.dataset) / BATCH_SIZE)
    effv_batches_per_epoch = math.ceil(len(train_loader.dataset) / EFFV_BATCH_SIZE)
    INPUT_IMG_KEY = data.SegDataset.IN_CROPPED_IMG if dset.has_bboxes else data.SegDataset.IN_ORIG_IMG
    TRUTH_IMG_KEY = data.SegDataset.GT_CROPPED_IMG if dset.has_bboxes else data.SegDataset.GT_ORIG_IMG

    for epoch in range(1, EPOCHS+1):
        sched.step()
        print("Current learning rate:", sched.get_lr())
        total_loss = 0
        ct = 0
        samples_per_epochs_past = (epoch - 1) * len(train_loader.dataset)
        updates_per_epochs_past = (epoch - 1) * effv_batches_per_epoch
        train_loader = set_loader_dset(train_loader, train=True)

        effv_ct = 0
        sum_loss = 0

        for i, sample in enumerate(train_loader):
            input = sample[INPUT_IMG_KEY]
            ct += input.size(0)
            effv_ct += input.size(0)
            truth = sample[TRUTH_IMG_KEY]
            input, truth = input.to(DEV), truth.to(DEV)

            output = model(input)
            output = output.squeeze(1)

            loss = criterion(output, truth)
            sum_loss += loss
            total_loss += loss.detach()

            if effv_ct == EFFV_BATCH_SIZE:
                avg_loss = sum_loss / effv_ct
                avg_loss.backward()
                optim.step()
                optim.zero_grad()
                effv_ct = 0
                sum_loss = 0

            if (updates_per_epochs_past + i) % UPDATE_EVERY == 0:
                samples = samples_per_epochs_past + ct
                if KFOLD != 0:
                    sampler.train = False
                validate(model,
                         valid_loader,
                         criterion,
                         f"s{samples}_e{epoch:03d}_u{i:03d}.png",
                         writer=valid_writer,
                         step=samples)
                model.train()
                if KFOLD != 0:
                    sampler.train= True
                train_loader = set_loader_dset(train_loader, train=True)

        if KFOLD != 0:
            sampler.next_fold()
            print(f"Validation fold set to {sampler.i}")

        if effv_ct != 0:
            avg_loss = sum_loss / effv_ct
            avg_loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"finished epoch {epoch}")
        samples = samples_per_epochs_past + ct
        train_writer.add_scalar(f"data/{criterion.__name__}",
                                float(total_loss) / ct, samples)

    train_writer.close()
    valid_writer.close()


if __name__ == "__main__":
    try:
        os.makedirs(OUTPUT_PATH)
    except:
        print(f"ok to remove {OUTPUT_PATH}? ENTER or CTRL-C")
        input("> ")
        shutil.rmtree(OUTPUT_PATH)
        os.makedirs(OUTPUT_PATH)

    if os.path.exists(LOGDIR):
        print(f"ok to remove {LOGDIR}? ENTER or CTRL-C")
        input("> ")
        shutil.rmtree(LOGDIR)

    main()
