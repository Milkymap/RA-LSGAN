import torch as th 
from torch.utils.data import Dataset 

from os import path
from PIL import Image  
from glob import glob 

class DataHolder(Dataset):
	def __init__(self, data_path, extension='*', mapper=None):
		super(DataHolder, self).__init__()
		self.files = glob(path.join(data_path, extension))[:2048]
		self.mapper = mapper 

	def __normalize(self, image):
		if self.mapper is not None:
			return self.mapper(image)
		return image 

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		image = Image.open(self.files[idx])
		return self.__normalize(image)
