import os
import cv2
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from PIL import Image


class COAD(data_utils.Dataset):
    ''' Unlabeled Dataset Class'''
    def __init__(self, data_path, exclude=None, mode='20.0', limit=float("inf"), transform=None):
        super().__init__()
        assert mode in ['40.0', '20.0', '10.0', '5.0']

        self.data_path = data_path 
        self.indexed_foldername = [] 
        self.indexed_filename = [] 
        
        if transform is not None:
            self.transform = transform 
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            
        if exclude is not None:
            patient_id = [file_name.split('-')[2] for file_name in exclude]
        else:
            patient_id = []
        
        i = 0
        for folder in os.listdir(data_path):
            i += 1
            if i < limit and folder[-5:] == 'files':
                folder = folder + '/' + mode 
                patch_path = os.path.join(data_path, folder)
                if os.path.isdir(patch_path):
                    for file in os.listdir(patch_path):
                        if folder.split('-')[2] not in patient_id:
                            self.indexed_foldername.append(folder)
                            self.indexed_filename.append(file)
                      
    def __len__(self):
        i = 0
        for file in self.indexed_filename:
            i +=1
        return i

    def __getitem__(self, idx):
        image_filepath = self.data_path + '/' + self.indexed_foldername[idx] + '/' + self.indexed_filename[idx]
        image = Image.open(image_filepath)
        image = self.transform(image)
        label = 0    
        return image, label