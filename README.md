# skin-lesion

Obtain all the ISIC 2018 skin lesion data from [the challenge website](https://challenge2018.isic-archive.com/) or [kitware](https://challenge.kitware.com/#challenge/5aab46f156357d5e82b00fe5).

## Install

The recommended install uses ``conda`` and an internet connection. Install [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://conda.io/miniconda.html) (Python 3 install recommended but not strictly necessary). Then install our conda environment with:
```
conda env create -f requirements.yml -n isic2018
```

To activate the environment, run:
```
conda activate isic2018
```
On \*nix with older versions of conda, you may need to use ``. activate isic2018`` or ``source activate isic2018`` instead.

## Working with the code

The scripts will always use CLI arguments for paths. You can set the ``ISIC`` environment variable to the folder containing the unzipped data folders, and keep the challenge's names for the folders. Then the defaults for the CLI args will be correct and you won't have to specify them.

## FAQ
Q: I downloaded the data. How can I unzip it all conveniently? (with bash)?

A: 
```bash
for i in $( ls *.zip ); do unzip $i; done
for i in $( ls *.zip ); do rm $i; done
```

Q: What's the ISIC data's license?

A: It's Creative COmmons license. It is not licensed with (nor included with) this software.

