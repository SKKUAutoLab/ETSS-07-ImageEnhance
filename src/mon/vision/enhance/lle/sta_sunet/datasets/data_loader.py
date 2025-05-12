import os
import torch
from torchvision import transforms
from datasets.data_augment import RandomCrop, RandomFlip, ToTensor
from torch.utils.data import Dataset
import cv2
import glob
import re
import numpy as np


class BVILowLight:
    
    def __init__(self, config):
        self.config = config
        
    def load_lowlight(self, topatch=True):
        print("=> load LLdataset using dataset '{}'".format(self.config.dataset.type))
        train_dataset = LowLightDataset(
            self.config.dataset.train_file,
            train          = True,
            root_distorted = self.config.dataset.root_distorted,
            root_restored  = self.config.dataset.root_restored,
            network        = self.config.model.network,
            numframes      = self.config.dataset.num_frames,
            transform      = transforms.Compose([
                RandomCrop(output_size=self.config.dataset.image_size, topleft=self.config.dataset.aug_topleft),
                RandomFlip(),
                ToTensor(network=self.config.model.network)
            ])
        )
        
        if topatch:
            val_dataset = LowLightDataset(
                self.config.dataset.val_file,
                train          = False,
                root_distorted = self.config.dataset.root_distorted,
                root_restored  = self.config.dataset.root_restored,
                network        = self.config.model.network,
                numframes      = self.config.dataset.num_frames,
                transform      = transforms.Compose([
                    RandomCrop(output_size=self.config.dataset.image_size, topleft=self.config.dataset.aug_topleft),
                    ToTensor(network=self.config.model.network)
                ])
            )
        else:  # test
            val_dataset = LowLightDataset(
                self.config.dataset.val_file,
                train          = False,
                root_distorted = self.config.dataset.root_distorted,
                root_restored  = self.config.dataset.root_restored,
                network        = self.config.model.network,
                numframes      = self.config.dataset.num_frames,
                transform      = False
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size  = self.config.training.batch_size,
            shuffle     = True,
            num_workers = self.config.dataset.num_workers,
            pin_memory  = True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size  = 1,
            shuffle     = False,
            num_workers = self.config.dataset.num_workers,
            pin_memory  = True
        )

        return train_loader, val_loader


"""BVI-RLV dataset."""
class LowLightDataset(Dataset):
    
    @staticmethod
    def get_file_paths(root, folder_name, patterns, n): 
        all_files = []
        for pattern in patterns:
            full_pattern_path = os.path.join(root, folder_name, pattern)
            # find all files matching the current pattern
            matching_files = glob.glob(full_pattern_path)
            matching_files.sort() 

            # For multi-frame input only: exclude the first and last n frames in each scene
            if len(matching_files) > 2 * n:
                all_files.extend(matching_files[n:-n])
            else:
                all_files.extend([])
                print("Not enough files for multi-frame input")

        return all_files

    def __init__(self, train_file, train, root_distorted, root_restored, network="STASUNet", numframes=5, transform=None):
        self.root_distorted = root_distorted
        self.root_restored  = root_restored
        self.transform      = transform
        self.network        = network
        self.numframes      = numframes

        # Read folder names from the training txt file
        with open(train_file, "r") as file:
            self.folder_names = file.read().splitlines()
        # print("read training folder names: ", self.folder_names)

        # log restored and distorted image patterns
        restored_patterns  = [os.path.join("normal_light_10", "*.png"), os.path.join("normal_light_20", "*.png")]
        distorted_patterns = [os.path.join("low_light_10",    "*.png"), os.path.join("low_light_20",    "*.png")]

        self.filesnames     = [path for folder in self.folder_names for path in self.get_file_paths(self.root_restored,  folder, restored_patterns,  numframes//2)]
        self.distortednames = [path for folder in self.folder_names for path in self.get_file_paths(self.root_distorted, folder, distorted_patterns, numframes//2)]
        # print("input file number",len(self.filesnames))
        # print("gt file number",len(self.distortednames))
        
    def __len__(self):
        return len(self.distortednames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        input_path = self.distortednames[idx]
        gt_path    = self.filesnames[idx]
        input_root, input_filename = os.path.split(input_path)
        current_frame = int(re.search(r"(\d+).png", input_filename).group(1))
        halfcurf      = int(self.numframes / 2)  # middle frame
        first_frame   = True

        # construct files as multi-frame input
        for i in range(current_frame - halfcurf, current_frame + halfcurf + 1): 
            frame_number       = f"{i:05d}"  # zero-padded frame number
            new_input_filename = f"{frame_number}.png"
            new_input_path     = os.path.join(input_root, new_input_filename)
            # Load the input image
            input_frame = cv2.imread(new_input_path,cv2.IMREAD_COLOR)
            input_frame = input_frame.astype('float32')
            if self.network == "STASUNet":
                input_frame = input_frame[..., np.newaxis]

            if first_frame:
                image       = input_frame / 255.0
                first_frame = False 
            else: 
                if self.network == "STASUNet":
                    image = np.append(image, input_frame / 255.0, axis=3)
                else:  # PCDUNet
                    image = np.append(image, input_frame / 255.0, axis=2)
 
        groundtruth = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        groundtruth = groundtruth.astype("float32")
        groundtruth = groundtruth/255.
        
        light  = os.path.split(input_root)[-1]
        scene  = os.path.split(os.path.split(input_root)[0])[-1]
        img_id = light + "-" + scene + "-" + input_filename[:-4]

        sample = {"image": image, "groundtruth": groundtruth}
        if self.transform:
            sample = self.transform(sample)
            
        sample["img_id"] = img_id

        return sample
