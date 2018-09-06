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

from args import add_input, add_truth, add_model_dir
import data
from segnet import SegNet

CHW_TO_HWC = (1, 2, 0)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
ROI_CHANNEL = 1
NUM_SAMPLES = 4

# these need to become flags
TRAIN_WORKERS = 4
VALID_WORKERS = 4
EPOCHS = 25
BATCH_SIZE = 12
OUTPUT_PATH = os.path.join("..", "output")
LOGDIR = os.path.join("..", "tb")
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


class Grid:
    __slots__ = ["path", "fig", "axes", "n"]

    def __init__(self, path, n_samples):
        self.path = path
        self.fig, self.axes = plt.subplots(n_samples, 4)
        self.n = 0

    def save(self):
        self.fig.savefig(self.path)

    def close(self):
        plt.close(self.fig)

    def push(self, input, truth, output):
        # after spending an hour trying to figure out why sample_binry was
        # black, I've decided safety checks are worth the speed penalty.
        assert input.dim() == 3
        assert truth.dim() == 2
        assert output.dim() == 3
        assert output.shape[0] == NUM_CLASSES
        (inp, tru, out, tsh) = self.axes[self.n]
        sample_input = input.permute(*CHW_TO_HWC)
        sample_truth = truth
        sample_outpt = output[ROI_CHANNEL, ...]
        sample_binry = (F.softmax(output, 0)[ROI_CHANNEL, ...] >= 0.5)

        inp.imshow(sample_input)
        tru.imshow(sample_truth, cmap="Greys_r")
        out.imshow(sample_outpt, cmap="Greys_r")
        tsh.imshow(sample_binry, cmap="Greys_r")

        self.n += 1


def save_sample(path, input, truth, output):
    sample_input = input.permute(*CHW_TO_HWC).cpu().numpy()
    sample_truth = truth.cpu().numpy()
    sample_outpt = output[ROI_CHANNEL, ...].cpu().numpy()
    sample_binry = F.softmax(output[ROI_CHANNEL, ...], 1) >= 0.5

    fig, (inp, tru, out, tsh) = plt.subplots(1, 4)
    inp.imshow(sample_input)
    tru.imshow(sample_truth, cmap="Greys_r")
    out.imshow(sample_outpt, cmap="Greys_r")
    tsh.imshow(sample_binry, cmap="Greys_r")
    fig.savefig(path)


def validate(model, valid_loader, criterion, name, writer=None, step=None):
    model.eval()
    total_xent = 0
    ct = 0
    random_sample = set(random.sample(range(len(valid_loader.dataset)), NUM_SAMPLES))
    grid = Grid(os.path.join(OUTPUT_PATH, name), NUM_SAMPLES)
    with torch.no_grad():
        for i, sample in enumerate(valid_loader):
            input = sample[data.SegDataset.INPUT_IMG]
            ct += input.shape[0]
            truth = sample[data.SegDataset.TRUTH_IMG]
            input, truth = input.to(DEV), truth.to(DEV)

            output = model(input)

            loss = criterion(output, truth)
            total_xent += float(loss)

            for x in range((i * BATCH_SIZE), (i+1) * BATCH_SIZE):
                if x in random_sample:
                    grid.push(input[x % BATCH_SIZE],
                              truth[x % BATCH_SIZE],
                              output[x % BATCH_SIZE])

        if writer is not None:
            assert step is not None
            writer.add_scalar("data/xentropy", total_xent / ct, step)

        grid.save()
        grid.close()
        print("avg validation xent:", total_xent / ct)


def main():
    parser = argparse.ArgumentParser()
    add_input(parser, cropped=True)
    add_truth(parser, cropped=True)
    add_model_dir(parser)
    args = parser.parse_args()

    dset = data.SegDataset(args.input, args.truth, xform=data.torch_xform)
    train_size = int(round(len(dset) * 0.8))
    valid_size = len(dset) - train_size
    dset_train, dset_valid = random_split(dset, (train_size, valid_size))
    print(f"Training with {len(dset_train)} samples")
    print(f"Validating with {len(dset_valid)} samples")
    model = SegNet(NUM_CLASSES, pretrained=True, model_dir=args.model_dir)
    print(model)
    model = model.to(DEV)
    if DEV.type == "cuda":
        print("Using GPU")
    model.train()
    train_loader = DataLoader(
        dset_train, batch_size=BATCH_SIZE, num_workers=TRAIN_WORKERS, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(
        dset_valid, batch_size=BATCH_SIZE, num_workers=VALID_WORKERS, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    UPDATE_EVERY = 100

    valid_writer = SummaryWriter(log_dir=os.path.join(LOGDIR, "valid"))
    train_writer = SummaryWriter(log_dir=os.path.join(LOGDIR, "train"))

    batches_per_epoch = math.ceil(len(train_loader.dataset) / BATCH_SIZE)
    for epoch in range(1, EPOCHS+1):
        total_xent = 0
        ct = 0
        steps_per_epochs_past = (epoch - 1) * len(train_loader.dataset)
        updates_per_epochs_past = (epoch - 1) * batches_per_epoch
        for i, sample in enumerate(train_loader):
            input = sample[data.SegDataset.INPUT_IMG]
            ct += input.size(0)
            truth = sample[data.SegDataset.TRUTH_IMG]
            input, truth = input.to(DEV), truth.to(DEV)

            output = model(input)

            loss = criterion(output, truth)
            total_xent += loss.detach()
            loss.backward()
            optim.step()

            if (updates_per_epochs_past + i) % UPDATE_EVERY == 0:
                step = steps_per_epochs_past + ct
                validate(model,
                         valid_loader,
                         criterion,
                         f"s{step}_e{epoch:03d}_u{i:03d}.png",
                         writer=valid_writer,
                         step=step)
                model.train()

        print(f"finished epoch {epoch}")
        step = steps_per_epochs_past + ct
        train_writer.add_scalar("data/xentropy", float(total_xent) / ct, step)

    train_writer.close()
    valid_writer.close()


if __name__ == "__main__":
    main()
