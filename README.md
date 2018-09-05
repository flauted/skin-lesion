# skin-lesion

Obtain all the ISIC 2018 skin lesion data from
[the challenge website](https://challenge2018.isic-archive.com/)
or
[kitware](https://challenge.kitware.com/#challenge/5aab46f156357d5e82b00fe5).

## Install

The recommended install uses ``conda`` and an internet connection. Install
[Anaconda](https://www.anaconda.com/download/#linux)
or
[Miniconda](https://conda.io/miniconda.html)
(Python 3 install recommended but not strictly necessary).
Then install our conda environment with:
```bash
conda env create -f requirements.yml -n isic2018
```
if you have a GPU and with:
```bash
conda env create -f requirements_cpu.yml -n isic2018
```
if you do not have a GPU.

To activate the environment, run:
```bash
conda activate isic2018
```
On \*nix with older versions of conda, you may need to use
``. activate isic2018``
or
``source activate isic2018`` instead.

## Working with the code

The scripts will always use CLI arguments for paths.

For data paths, you can set the ``ISIC`` environment variable to the folder containing the unzipped data folders,
and keep the challenge's names for the folders.
Then the defaults for the CLI args will be correct and you won't have to specify them.

For pretrained model paths, the default location is 

> ``$TORCH_HOME/models`` where ``$TORCH_HOME`` defaults to ``~/.torch``. The default
> directory can be overridden with the ``$TORCH_MODEL_ZOO`` environment variable.

That is consistent with
[``torch.utils.model_zoo``](https://pytorch.org/docs/stable/model_zoo.html?highlight=model_zoo#module-torch.utils.model_zoo)

### Preprocessing
Run ``resize_imgs.py`` to crop all images to a more reasonable size (maintaining aspect ratio).
This drastically reduces the runtime. 

### Visualization
After starting a training session, navigate to the root directory of this repo and use
``tensorboard --logdir LOGDIR`` where ``LOGDIR`` is the path to your tensorboard logs.
By default, this is the ``tb`` folder.

## FAQ
Q: I downloaded the data. How can I unzip it all conveniently? (with bash)?

A: 
```bash
for i in $( ls *.zip ); do unzip $i; done
for i in $( ls *.zip ); do rm $i; done
```

Q: What's the ISIC data's license?

A: It's Creative Commons license. It is not licensed with (nor included with) this software.

Q: SegNet references?

A: [clean implementation](https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/seg_net.py),
[dirty (but explicit) implementation](https://github.com/delta-onera/delta_tb/blob/master/semantic_segmentation/model/segnet.py),
and [the paper](https://arxiv.org/pdf/1511.00561.pdf)

Q: Why do I need TensorFlow?

A: Technically you just need TensorBoard but it depends on TensorFlow.
We're using TensorboardX for training visualization and that requires TensorBoard. 
You're welcome to try installing TensorBoard without TensorFlow and get it to work with TensorboardX.
If you do, please let me know!

Q: Do I need GPU TensorFlow?

A: Nope. Tensorboard is all that matters, so any version of TensorFlow should work.
