"""Keep CLI documentation/flags/defaults consistent by using these."""
import os
import default_paths as dpaths


def add_input(parser, cropped=False, name=("-i", "--input")):
    if cropped:
        parser.add_argument(name[0], name[1], type=str,
                            default=os.path.join(dpaths.isic_default, dpaths.task_12_training_cropped))
    else:
        parser.add_argument(name[0], name[1], type=str,
                            default=os.path.join(dpaths.isic_default, dpaths.task_12_training))


def add_truth(parser, cropped=False, name=("-t", "--truth")):
    if cropped:
        parser.add_argument(name[0], name[1], type=str,
                            default=os.path.join(dpaths.isic_default, dpaths.task_1_training_gt_cropped))
    else:
        parser.add_argument(name[0], name[1], type=str,
                            default=os.path.join(dpaths.isic_default, dpaths.task_1_training_gt))


def add_model_dir(parser):
    parser.add_argument("-m", "--model-dir", default=dpaths.DEFAULT_MODEL_DIR,
                        help="Location of models parent directory "
                             "(defaults to $TORCH_MODEL_ZOO, which defaults to $TORCH_HOME/models. "
                             "$TORCH_HOME defaults to ~/.torch")
