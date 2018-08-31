import os
try:
    isic_default = os.environ["ISIC"]
except KeyError:
    isic_default = os.getcwd()


task_12_training = "ISIC2018_Task1-2_Training_Input"
task_1_training_gt = "ISIC2018_Task1_Training_GroundTruth"

