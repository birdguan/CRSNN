import os
import sys
sys.path.append(".")
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from feature_extraction_unit.SpykeTorch import snn
from feature_extraction_unit.SpykeTorch import functional as sf
from feature_extraction_unit.SpykeTorch import visualization as vis
from feature_extraction_unit.SpykeTorch import utils
from torchvision import transforms
import struct
import glob
import time
import matplotlib.pyplot as plt

use_cuda = True


class FeatureExtractionModel(nn.Module):
	def __init__(self):
		super(FeatureExtractionModel, self).__init__()

		self.conv1 = snn.Convolution(6, 30, 5, 0.8, 0.05)
		self.conv1_t = 15
		self.k1 = 5
		self.r1 = 3

		self.conv2 = snn.Convolution(30, 50, 3, 0.8, 0.05)
		self.conv2_t = 10
		self.k2 = 8
		self.r2 = 2

		self.conv3 = snn.Convolution(50, 40, 5, 0.8, 0.05)
		self.conv3_t = 23.8

		self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
		self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
		self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
		self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
		self.max_ap = Parameter(torch.Tensor([0.15]))

		self.decision_map = []
		for i in range(10):
			self.decision_map.extend([i] * 20)

		self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
		self.spk_cnt1 = 0
		self.spk_cnt2 = 0
		self.potential1 = None
		self.spike1 = None
		self.potential2 = None
		self.spike2 = None
		self.spike3 = None
		self.potential3 = None
		self.winners3 = None

	def forward(self, input, max_layer):
		input = sf.pad(input, (2, 2, 2, 2), 0)
		if self.training:
			potential = self.conv1(input)
			spike, potential = sf.fire(potential, self.conv1_t, True)
			if max_layer == 1:
				self.spk_cnt1 += 1
				if self.spk_cnt1 >= 500:
					self.spk_cnt1 = 0
					ap = torch.tensor(self.stdp1.learning_rate[0][0].item(),
									  device=self.stdp1.learning_rate[0][0].device) * 2
					ap = torch.min(ap, self.max_ap)
					an = ap * -0.75
					self.stdp1.update_all_learning_rate(ap.item(), an.item())
				potential = sf.pointwise_inhibition(potential)
				spike = potential.sign()
				winners = sf.get_k_winners(potential, self.k1, self.r1, spike)
				self.ctx["input_spikes"] = input
				self.ctx["potentials"] = potential
				self.ctx["output_spikes"] = spike
				self.ctx["winners"] = winners
				return spike, potential
			spk_in = sf.pad(sf.pooling(spike, 2, 2), (1, 1, 1, 1))
			potential = self.conv2(spk_in)
			spike, potential = sf.fire(potential, self.conv2_t, True)
			if max_layer == 2:
				self.spk_cnt2 += 1
				if self.spk_cnt2 >= 500:
					self.spk_cnt2 = 0
					ap = torch.tensor(self.stdp2.learning_rate[0][0].item(),
									  device=self.stdp2.learning_rate[0][0].device) * 2
					ap = torch.min(ap, self.max_ap)
					an = ap * -0.75
					self.stdp2.update_all_learning_rate(ap.item(), an.item())
				potential = sf.pointwise_inhibition(potential)
				spike = potential.sign()
				winners = sf.get_k_winners(potential, self.k2, self.r2, spike)
				self.ctx["input_spikes"] = spk_in
				self.ctx["potentials"] = potential
				self.ctx["output_spikes"] = spike
				self.ctx["winners"] = winners
				return spike, potential
			spk_in = sf.pad(sf.pooling(spike, 3, 3), (2, 2, 2, 2))
			potential = self.conv3(spk_in)
			spike = sf.fire(potential)
			winners = sf.get_k_winners(potential, 1, 0, spike)
			self.ctx["input_spikes"] = spk_in
			self.ctx["potentials"] = potential
			self.ctx["output_spikes"] = spike
			self.ctx["winners"] = winners
			output = -1
			if len(winners) != 0:
				output = self.decision_map[winners[0][0]]
			return output
		else:
			potential = self.conv1(input)
			spike, potential = sf.fire(potential, self.conv1_t, True)
			self.potential1 = potential
			self.spike1 = spike
			if max_layer == 1:
				return spike, potential
			potential = self.conv2(sf.pad(sf.pooling(spike, 2, 2), (1, 1, 1, 1)))
			spike, potential = sf.fire(potential, self.conv2_t, True)
			self.spike2 = spike
			self.potential2 = potential
			if max_layer == 2:
				return spike, potential
			potential = self.conv3(sf.pad(sf.pooling(spike, 3, 3), (2, 2, 2, 2)))
			spike = sf.fire(potential, self.conv3_t)
			self.potential3 = potential
			self.spike3 = spike
			winners = sf.get_k_winners(potential, 1, 0, spike)
			self.winners3 = winners
			output = -1
			if len(winners) != 0:
				output = self.decision_map[winners[0][0]]
			return output

	def get_potential_and_spike(self):
		out_info = {}
		out_info["potential1"] = self.potential1
		out_info["potential2"] = self.potential2
		out_info["potential3"] = self.potential3
		out_info["spike1"] = self.spike1
		out_info["spike2"] = self.spike2
		out_info["spike3"] = self.spike3
		out_info["winners3"] = self.winners3
		return out_info

	def stdp(self, layer_idx):
		if layer_idx == 1:
			self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
		if layer_idx == 2:
			self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])


def train_unsupervise(network, data, layer_idx):
	network.train()
	for i in range(len(data)):
		data_in = data[i]
		if use_cuda:
			data_in = data_in.cuda()
		network(data_in, layer_idx)
		network.stdp(layer_idx)


def stochastic_decay_gpu(cascade_tensor, new_tensor, stochastic_prob, decay_rate):
	data = torch.tensor(new_tensor.size())
	new_cascade_tensor = torch.tensor(cascade_tensor.size())
	for i in range(5):
		random_tensor = torch.rand(new_tensor.size())
		random_tensor = torch.where(random_tensor < stochastic_prob)
		cascade_tensor[i+1] = cascade_tensor[i+1] - cascade_tensor[i+1].multiple(random_tensor).\
			multiple(torch.ones(new_tensor.size()*decay_rate))
		new_cascade_tensor[i] = cascade_tensor[i+1]
	new_cascade_tensor[5] = new_tensor
	for i in range(6):
		data[i] = new_cascade_tensor[i, i]
	return data, new_cascade_tensor


class TemporalFilter(filter):
	def __init__(self, filter, timesteps=15):
		self.to_tensor = transforms.ToTensor()
		self.filter = filter
		self.temporal_transform = utils.Intensity2Latency(timesteps)
		self.cnt = 0

	def __call__(self, image):
		if self.cnt % 1000 == 0:
			print(self.cnt)
		self.cnt += 1
		image = self.to_tensor(image) * 255
		image.unsqueeze_(0)
		image = self.filter(image)
		image = sf.local_normalization(image, 8)
		temporal_image = self.temporal_transform(image)
		return temporal_image.sign()


kernels = [utils.DoGKernel(3, 3 / 9, 6 / 9),
			utils.DoGKernel(3, 6 / 9, 3 / 9),
			utils.DoGKernel(7, 7 / 9, 14 / 9),
			utils.DoGKernel(7, 14 / 9, 7 / 9),
			utils.DoGKernel(13, 13 / 9, 26 / 9),
			utils.DoGKernel(13, 26 / 9, 13 / 9)]
filter = utils.Filter(kernels, padding=6, thresholds=50)
temporalFilterCascade = TemporalFilter(filter)


class FeatureExtractionUnit():
	def __init__(self):
		self.model = FeatureExtractionModel()
		self.model.cuda()
		blank_image = np.zeros([32, 32])
		self.cascade_tensor = torch.tensor(np.array([blank_image for _ in range(6)]))
		if os.path.isfile("layer1_model.net"):
			self.model.load_state_dict(torch.load("layer1_model.net"))
		else:
			print("[WARNING] \"layer1_model.net\" does not exist")
		if os.path.isfile("layer2_mode.net"):
			self.model.load_state_dict(torch.load("layer2_model.net"))
		else:
			print("[WARNING] \"layer2_model.net\" does not exist")
		if os.path.isfile("layer3_model.net"):
			self.model.load_state_dict(torch.load("layer3_model.net"))
		else:
			print("[WARNING] \"layer3_model.net\" does not exist")
		self.model.eval()

	def getLaneFeature(self, laneData):
		laneDataTensor = laneData.tensor()
		laneData, self.cascade_tensor = stochastic_decay_gpu(self.cascade_tensor, laneDataTensor, stochastic_prob=0.2, decay_rate=0.1)
		data_in = laneDataTensor
		d = self.model(data_in, 3)
		out_info = self.model.get_potential_and_spike()
		spike3 = out_info["spike3"].cpu().numpy()[-1]
		potential3 = out_info["potential3"].cpu().numpy()[-1]
		feature_potential = np.zeros((40, 4, 4), dtype=np.float32)
		for i in range(40):
			for m in range(2):
				for n in range(2):
					if spike3[i*5, m, n] != 0:
						feature_potential[i, m, n] = potential3[i, m, n]
					else:
						feature_potential[i, m, n] = 0
		return feature_potential.flatten()
