import click 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import torch.optim as optim
from torch.utils.data import DataLoader 

from libraries.strategies import * 
from libraries.log import logger  
from models.generator import Generator 
from models.discriminator import Discriminator
from optimization.dataholder import DataHolder 

from torchvision import transforms as T 

from os import path, mkdir 

@click.command()
@click.option('-p', '--data_path', help='path to training data', type=click.Path(True))
@click.option('-e', '--nb_epochs', help='number of epochs', type=int)
@click.option('-b', '--bt_size', help='batch size', type=int)
@click.option('-s', '--storage', help='image sampler storage', type=click.Path(False))
def main_loop(data_path, nb_epochs, bt_size, storage):
	device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
	if not path.isdir(storage):
		mkdir(storage)

	G = Generator(4).to(device)
	D = Discriminator(4).to(device)

	adv_criterion = nn.MSELoss().to(device)
	gen_solver = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
	dis_solver = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

	mapper = T.Compose([	
		T.Resize((128, 128)), 
		T.ToTensor(), 
		T.Normalize([0.5] * 3, [0.5] * 3)]
	)

	data_holder = DataHolder(data_path, '*.jpg', mapper)
	data_loader = DataLoader(data_holder, shuffle=True, batch_size=bt_size, drop_last=True)

	for epoch_counter in range(nb_epochs):
		for index, image in enumerate(data_loader):
			real_image = image.to(device)

			real_label = th.ones(real_image.size(0), 1).to(device) 
			fake_label = th.zeros(real_image.size(0), 1).to(device)

			# ... generator ... 

			gen_solver.zero_grad()
			
			fake_noise = th.randn(image.size(0), 128, 1, 1).to(device)
			fake_image = G(fake_noise)
			
			D_F = D(fake_image)
			D_R = D(real_image).detach()
			
			LG0 = adv_criterion(D_F - th.mean(D_R, dim=0, keepdim=True), real_label)
			LG1 = adv_criterion(D_R - th.mean(D_F, dim=0, keepdim=True), fake_label)
			LG2 = LG0 + LG1  

			LG2.backward()
			
			gen_solver.step()

			# ... discriminator ... 

			dis_solver.zero_grad()

			D_F = D(fake_image.detach())
			D_R = D(real_image)
			
			LD0 = adv_criterion(D_R - th.mean(D_F, dim=0, keepdim=True), real_label)
			LD1 = adv_criterion(D_F - th.mean(D_R, dim=0, keepdim=True), fake_label)
			LD2 = (LD0 + LD1) / 2 

			LD2.backward()
			dis_solver.step()

			MSG = (nb_epochs, epoch_counter, index, LG2.item(), LD2.item())
			logger.debug('[%03d/%03d]:%05d >> EG : %07.3f | ED : %07.3F' % MSG)

			if index % 2 == 0:
				sample_image = to_grid(th.cat((fake_image.cpu(), real_image.cpu()), dim=-1), nb_rows=2)
				cv_version = th2cv(sample_image)
				rescaled_version = cv_version
				cv2.imshow('sample', rescaled_version) 
				cv2.waitKey(10)

		# end current epoch
	# end training loop 

if __name__ == '__main__':
	logger.debug(' ... [training] ... ') 
	main_loop()