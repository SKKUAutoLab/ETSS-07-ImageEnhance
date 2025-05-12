import glob
import os

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class BaseDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super().__init__()

    def initialize(self, args, task):
        pass

    def extract_number(self, filename):
        match = os.path.splitext(os.path.split(filename)[1])[0]
        return int(match) if match else 0  # Extract the numeric part and convert it to an integer

    def sort_files_by_name(self, img_list):
        # Sort the file
        sorted_files = sorted(img_list, key=self.extract_number)
        return sorted_files


class DefaultDataset(BaseDataset):
    
    def initialize(self, args, task):
        self.args        = args
        self.low_img_dir = args.lowlight_images_path
        self.task        = task
        self.train_low_data_names    = []
        self.train_target_data_names = []
        assert os.path.exists(self.low_img_dir), "Input directory does not exist!"

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                if "." == name[0]:
                    continue
                self.train_low_data_names.append(os.path.join(root, name))

        # self.train_low_data_names.sort()
        self.train_low_data_names = self.sort_files_by_name(self.train_low_data_names)
        # if 'test' == task:
        #     self.train_low_data_names = self.train_low_data_names[:10]

        self.count      = len(self.train_low_data_names)
        transform_list  = []
        transform_list += [transforms.ToTensor()]
        self.transform  = transforms.Compose(transform_list)
        self.last_data_name_path = self.train_low_data_names[0]

    def load_images_transform(self, file):
        im       = Image.open(file).convert("RGB")
        new_size = (1920, 1080)
        im       = im.resize(new_size)
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):
        ll       = self.load_images_transform(self.train_low_data_names[index])
        img_name = os.path.splitext(os.path.basename(self.train_low_data_names[index]))[0]
        img_path = self.train_low_data_names[index]
        last_data_name_path      = self.last_data_name_path
        self.last_data_name_path = img_path
        return ll, img_name, img_path, last_data_name_path

    def __len__(self):
        return self.count

    def name(self):
        return "Default"


class RLVDataLoader(BaseDataset):
    
    def initialize(self, args, task):
        self.args        = args
        self.low_img_dir = args.lowlight_images_path
        self.task        = task
        self.train_low_data_names    = []
        self.train_target_data_names = []
        assert os.path.exists(self.low_img_dir), "Input directory does not exist!"

        self.train_low_data_names = self.load_dataset_BVI(self.low_img_dir, task)

        self.count      = len(self.train_low_data_names)
        transform_list  = []
        transform_list += [transforms.ToTensor()]
        self.transform  = transforms.Compose(transform_list)
        self.last_data_name_path = self.train_low_data_names[0]

    def load_dataset_BVI(self, dir, task):
        img_list       = []
        ll_10_dir_name = "low_light_10"
        ll_20_dir_name = "low_light_20"
        assert task == "train" or task == "test", "Invalid phase: " + str(task)

        phase_list_file = str(task) + "_list.txt"
        # image_list_lowlight = []
        with open(os.path.join(dir, phase_list_file), "r") as file:
            phase_list = file.readlines()
            assert len(phase_list) > 0, "No input data."

        for folder_name in phase_list:
            folder_name = folder_name.strip()
            # train_ll_10_list = glob.glob(os.path.join(dir, "input", folder_name, ll_10_dir_name, "*.png"))
            # train_ll_20_list = glob.glob(os.path.join(dir, "input", folder_name, ll_20_dir_name, "*.png"))
            train_ll_10_list = glob.glob(os.path.join(dir, folder_name, ll_10_dir_name, "*.png"))
            train_ll_20_list = glob.glob(os.path.join(dir, folder_name, ll_20_dir_name, "*.png"))

            # sort file by name:
            train_ll_10_list = self.sort_files_by_name(train_ll_10_list)
            train_ll_20_list = self.sort_files_by_name(train_ll_20_list)

            img_list.extend(train_ll_10_list)
            img_list.extend(train_ll_20_list)

        return img_list

    def load_images_transform(self, file):
        im       = Image.open(file).convert('RGB')
        new_size = (1920, 1080)
        im       = im.resize(new_size)
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):
        ll       = self.load_images_transform(self.train_low_data_names[index])
        img_name = os.path.splitext(os.path.basename(self.train_low_data_names[index]))[0]
        img_path = self.train_low_data_names[index]
        last_data_name_path = self.last_data_name_path
        self.last_data_name_path = img_path

        return ll, img_name, img_path, last_data_name_path

    def __len__(self):
        return self.count

    def name(self):
        return "BVI-RLV"


class DidDataloader(BaseDataset):
    
    def initialize(self, args, task):
        self.args        = args
        self.low_img_dir = args.lowlight_images_path
        self.task        = task
        self.train_low_data_names    = []
        self.train_target_data_names = []
        assert os.path.exists(self.low_img_dir), "Input directory does not exist!"

        self.train_low_data_names = self.load_dataset(self.low_img_dir, task)

        self.count      = len(self.train_low_data_names)
        transform_list  = []
        transform_list += [transforms.ToTensor()]
        self.transform  = transforms.Compose(transform_list)
        self.last_data_name_path = self.train_low_data_names[0]

    def load_dataset(self, dir, task):
        img_list = []

        assert task == "train" or task == "test", "Invalid phase: " + str(task)

        phase_list_file = str(task) + "_list.txt"
        # image_list_lowlight = []
        with open(os.path.join(dir, phase_list_file), "r") as file:
            phase_list = file.readlines()
            assert len(phase_list) > 0, "No input data."

        for folder_name in phase_list:
            folder_name   = folder_name.strip()
            train_ll_list = glob.glob(os.path.join(dir, "input", folder_name, "*.jpg"))
            train_ll_list.extend(glob.glob(os.path.join(dir, "input", folder_name, "*.png")))

            # sort file by name:
            train_ll_list = self.sort_files_by_name(train_ll_list)

            img_list.extend(train_ll_list)

        return img_list

    def load_images_transform(self, file):
        im       = Image.open(file).convert("RGB")
        new_size = (1920, 1080)
        im       = im.resize(new_size)
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):
        ll       = self.load_images_transform(self.train_low_data_names[index])
        img_name = os.path.splitext(os.path.basename(self.train_low_data_names[index]))[0]
        img_path = self.train_low_data_names[index]
        last_data_name_path      = self.last_data_name_path
        self.last_data_name_path = img_path
        return ll, img_name, img_path, last_data_name_path

    def __len__(self):
        return self.count

    def name(self):
        return "DID"
