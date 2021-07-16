import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

class DISBlock(nn.Module):
	def __init__(self, i_channels, o_channels, k_size, stride, padding, normalize, activation):
		super(DISBlock, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(i_channels, o_channels, k_size, stride, padding), 
			nn.BatchNorm2d(o_channels) if normalize else nn.Identity(),
			activation
		)

	def forward(self, X):
		return self.body(X)

class Discriminator(nn.Module):
	def __init__(self, depth=4):
		super(Discriminator, self).__init__()
		self.head = DISBlock(3, 64, 4, 2, 1, True, nn.LeakyReLU(0.2))
		self.body = nn.Sequential(*[
			DISBlock(64 * 2 ** idx, 64 * 2 ** (idx + 1), 4, 2, 1, True, nn.LeakyReLU(0.2))
			for idx in range(depth)
		])
		self.tail = nn.Sequential(
			DISBlock(64 * 2 ** depth, 1, 4, 2, 1, False, nn.Identity()),
			nn.AdaptiveAvgPool2d(1), 
			nn.Flatten()
		)


	def forward(self, X):
		return self.tail(self.body(self.head(X)))


if __name__ == '__main__':
	D = Discriminator(4)
	X = th.randn((3, 3, 128, 128))
	Y = D(X)
	print(D)
	print(Y.shape)

