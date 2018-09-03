import argparse
import os
import shutil

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from args import add_input, add_truth, add_model_dir
import data
from segnet import SegNet

CHW_TO_HWC = (1, 2, 0)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
ROI_CHANNEL = 1

# these need to become flags
EPOCHS = 1
BATCH_SIZE = 12
OUTPUT_PATH = os.path.join("..", "output")
try:
    os.makedirs(OUTPUT_PATH)
except:
    print(f"ok to remove {OUTPUT_PATH}? ENTER or CTRL-C")
    input("> ")
    shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_input(parser)
    add_truth(parser)
    add_model_dir(parser)
    args = parser.parse_args()

    dset = data.SegDataset(args.input, args.truth, xform=data.torch_xform)
    model = SegNet(NUM_CLASSES, pretrained=True, model_dir=args.model_dir)
    print(model)
    model = model.to(DEV)
    if DEV.type == "cuda":
        print("Using GPU")
    model.train()
    loader = DataLoader(dset, batch_size=BATCH_SIZE)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    UPDATE_EVERY = 20

    for epoch in range(1, EPOCHS+1):
        for i, sample in enumerate(loader):
            input = sample[data.SegDataset.INPUT_IMG]
            truth = sample[data.SegDataset.TRUTH_IMG]
            input, truth = input.to(DEV), truth.to(DEV)

            output = model(input)

            loss = criterion(output, truth)
            loss.backward()
            optim.step()

            if i % UPDATE_EVERY == 0:
                sample_input = input[0, ...].permute(*CHW_TO_HWC).cpu().numpy()
                sample_truth = truth[0, ...].cpu().numpy()
                sample_outpt = output[0, ROI_CHANNEL, ...].detach().cpu().numpy()

                fig, (inp, tru, out) = plt.subplots(1, 3)
                inp.imshow(sample_input)
                tru.imshow(sample_truth, cmap="Greys_r")
                out.imshow(sample_outpt, cmap="Greys_r")
                fig.savefig(os.path.join(OUTPUT_PATH, f"{epoch:03d}_{i:03d}.png"))

        print(f"finished epoch {epoch}")
