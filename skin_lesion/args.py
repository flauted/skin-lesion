"""Keep CLI documentation/flags/defaults consistent by using these."""
import os
from default_paths import isic_default, task_12_training, task_1_training_gt, DEFAULT_MODEL_DIR


def add_input(parser):
    parser.add_argument("-i", "--input", type=str, default=os.path.join(isic_default, task_12_training))


def add_truth(parser):
    parser.add_argument("-t", "--truth", type=str, default=os.path.join(isic_default, task_1_training_gt))


def add_model_dir(parser):
    parser.add_argument("-m", "--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Location of models parent directory "
                             "(defaults to $TORCH_MODEL_ZOO, which defaults to $TORCH_HOME/models. "
                             "$TORCH_HOME defaults to ~/.torch")
