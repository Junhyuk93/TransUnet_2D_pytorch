import sys
import torch
import argparse
import os


# Additional Scripts
from train import TrainTestPipe
from inference import SegInference
# from inferencefolder import SegInference

def main_pipeline(parser):
    device = 'cpu:0'
    if torch.cuda.is_available():
        device = 'cuda:0'

    if parser.mode == 'train':
        ttp = TrainTestPipe(train_path=parser.train_path,
                            test_path=parser.test_path,
                            model_path=parser.model_path,
                            device=device)

        ttp.train()

    # elif parser.mode == 'inference':
#         inf = SegInference(model_path=parser.model_path,
#                            device=device)

#         _ = inf.infer(parser.image_path)

#     elif parser.mode == 'inference':
#         inf = SegInference(model_path=parser.model_path,
#                            device=device)

#         # If --image_path is provided, infer a single image
#         if parser.image_path:
#             _ = inf.infer(parser.image_path)
#         # If --image_folder is provided, infer all images in the folder
#         elif parser.image_folder:
#             image_folder = parser.image_folder
#             image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
            
#             for image_file in image_files:
#                 image_path = os.path.join(image_folder, image_file)
#                 _ = inf.infer(image_path)
            
    elif parser.mode == 'inference':
        inf = SegInference(model_path=parser.model_path, device=device)

        if parser.image_path:
            _ = inf.infer(path=parser.image_path)
        elif parser.image_folder:
            _ = inf.infer(path=parser.image_folder)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'])
    parser.add_argument('--model_path', required=True, type=str, default=None)

    parser.add_argument('--train_path', required='train' in sys.argv,  type=str, default=None)
    parser.add_argument('--test_path', required='train' in sys.argv, type=str, default=None)

    # parser.add_argument('--image_path', required='infer' in sys.argv, type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default=None)
    parser = parser.parse_args()

    main_pipeline(parser)
