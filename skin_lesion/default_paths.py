import os
try:
    isic_default = os.environ["ISIC"]
except KeyError:
    isic_default = os.getcwd()

task_12_training = "ISIC2018_Task1-2_Training_Input"
task_12_training_cropped = "ISIC2018_Task1-2_Training_Input_Cropped"
task_1_training_gt = "ISIC2018_Task1_Training_GroundTruth"
task_1_training_gt_cropped = "ISIC2018_Task1_Training_GroundTruth_Cropped"

task_12_training_split_train = os.path.join("images", "train2018")
task_1_training_split_gt = "masks"
task_12_training_split_val = os.path.join("images", "val2018")

train_bbox = "bbox_isic_2018_train_results.json"
valid_bbox = "bbox_isic_2018_val_results.json"

DEFAULT_IGNORE_FILES = {"LICENSE.txt", "ATTRIBUTION.txt"}

DEFAULT_MODEL_DIR = None  # see args.add_model_dir or download for some explanation
