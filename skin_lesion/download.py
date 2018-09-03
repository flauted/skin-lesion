import argparse

import torchvision

import models
from args import add_model_dir
from default_paths import DEFAULT_MODEL_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_dir(parser)
    parser.add_argument("-a", "--architecture", default="vgg11_bn",
                        help="Anything in torchvision.models. See "
                             "https://pytorch.org/docs/stable/torchvision/models.html")
    args = parser.parse_args()

    try:
        try:
            ModelClass = getattr(models, args.architecture)
            can_specify_dir = True
        except AttributeError:
            ModelClass = getattr(torchvision.models, args.architecture)
            can_specify_dir = False
    except Exception as e:
        print(f"Failed to find architecture '{args.architecture}'")
        raise e

    kwargs = {"pretrained": True, "model_dir": args.model_dir}
    if not can_specify_dir and args.model_dir != DEFAULT_MODEL_DIR:
        print("{args.architecture} was found in torchvision, "
              "but the model_dir cannot be specified. "
              "Is it okay to fall back to the default model_dir?\n"
              "(Any ENTER to continue, CTRL-C to abort)")
        input("> ")

    if not can_specify_dir:
        del kwargs["model_dir"]

    try:
        model = ModelClass(**kwargs)
    except Exception as e:
        print("Failed to (down)load model")
        raise e

    print(model)
    print("Success!")
