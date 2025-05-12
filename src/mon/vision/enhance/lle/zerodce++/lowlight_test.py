import glob
import os
import time

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import model


def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	data_lowlight = Image.open(image_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool(scale_factor).cuda()
	DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))
	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)

	end_time = (time.time() - start)

	print(end_time)
	image_path = image_path.replace('test_data','result_Zero_DCE++')

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	with torch.no_grad():

		filePath = 'data/test_data/'	
		file_list = os.listdir(filePath)
		sum_time = 0
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
		    for image in test_list:

				print(image)
				sum_time = sum_time + lowlight(image)

		print(sum_time)
