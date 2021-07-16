import numpy as np 
import operator as op 
import itertools as it, functools as ft

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

class GENBlock(nn.Module):
	def __init__(self, i_channels, o_channels, k_size, stride, padding, normalize, activation):
		super(GENBlock, self).__init__()
		self.body = nn.Sequential(
			nn.ConvTranspose2d(i_channels, o_channels, k_size, stride, padding), 
			nn.BatchNorm2d(o_channels) if normalize else nn.Identity(),
			activation
		)

	def forward(self, X):
		return self.body(X)

class Generator(nn.Module):
	def __init__(self, depth):
		super(Generator, self).__init__()
		self.head = GENBlock(128, 1024, 4, 1, 0, True, nn.ReLU())
		self.body = nn.Sequential(*[
			GENBlock(1024 // (2 ** idx), 1024 // (2 ** (idx + 1)), 4, 2, 1, True, nn.ReLU())
			for idx in range(depth)
		])
		self.tail = GENBlock(1024 // (2 ** depth), 3, 4, 2, 1, False, nn.Tanh())

	def forward(self, X):
		return self.tail(self.body(self.head(X)))


if __name__ == '__main__':
	G = Generator(4)
	print(G)
	X = th.randn((3, 128, 1, 1))
	Y = G(X)
	print(Y.shape)
