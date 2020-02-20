#!/usr/bin/env python
from __future__ import print_function  

#from minisom import MiniSom

#from copy import deepcopy
import h5py
import cv2
from models import Models
from intrinsic_motivation import IntrinsicMotivation

from plots import plot_learning_progress, plot_log_goal_inv, plot_log_goal_fwd, plot_simple, plot_learning_comparisons

# from cv_bridge import CvBridge, CvBridgeError
import random
import os
import shutil
import pickle
import gzip
import datetime
import numpy as np
import signal
import sys, getopt
from utils import clamp_x, clamp_y, x_lims, y_lims, z_lims, speed_lim, Position, load_data
import threading
import random
from cam_sim import Cam_sim
from parameters import Parameters

from doepy import build, read_write # pip install doepy - it may require also diversipy

import tensorflow as tf

GPU_FRACTION = 0.7

if tf.__version__ < "1.8.0":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
    session = tf.Session(config=config)
else:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
    session = tf.compat.v1.Session(config=config)


class GoalBabbling():

	def __init__(self):

		# this simulates cameras and positions
		self.cam_sim = Cam_sim("../romi_data/rgb_rectified")
		self.parameters = Parameters()

		self.lock = threading.Lock()
		signal.signal(signal.SIGINT, self.Exit_call)

		print('Loading test dataset ', self.parameters.get('romi_dataset_pkl'))
		self.train_images, self.test_images, self.train_cmds, self.test_cmds, self.train_pos, self.test_pos = load_data(
			self.parameters.get('romi_dataset_pkl'), self.parameters.get('image_size'), step=self.parameters.get('romi_test_data_step'))


	def initialise(self, param):

		self.intrinsic_motivation = IntrinsicMotivation(param)
		self.models = Models(param)

		self.exp_iteration = param.get('exp_iteration')
		self.iteration = 0

		self.pos = []
		self.cmd = []
		self.img = []
		
		self.goal_code = []

		self.samples_pos=[]
		self.samples_img=[]
		self.samples_codes=[]
		self.test_positions=[]

		self.move = False

		self.current_goal_x = -1
		self.current_goal_y = -1
		self.current_goal_idx = -1
#		self.goal_db = ['./sample_images/img_0.jpg','./sample_images/img_200.jpg','./sample_images/img_400.jpg','./sample_images/img_600.jpg','./sample_images/img_800.jpg','./sample_images/img_1000.jpg','./sample_images/img_1200.jpg','./sample_images/img_1400.jpg','./sample_images/img_1600.jpg' ]
		self.goal_image = np.zeros((1, self.image_size, self.image_size, channels), np.float32)	

		self.count = 1

		np.random.seed() # change the seed

		self.prev_pos=self.get_starting_pos()


	def log_current_inv_mse(self, param):
		img_codes = self.models.encoder.predict(self.test_images[0: param.get('romi_test_size')])
		motor_pred = self.models.inv_model.predict(img_codes)
		mse = (np.linalg.norm(motor_pred-self.test_pos[0:param.get('romi_test_size')]) ** 2) / param.get('romi_test_size')
		print ('Current mse inverse code model: ', mse)
		self.models.logger_inv.store_log(mse)

	def log_current_fwd_mse(self):
		img_obs_code = self.models.encoder.predict(self.test_images[0:param.get('romi_test_size')])
		img_pred_code = self.models.fwd_model.predict(self.test_pos[0:param.get('romi_test_size')])
		mse = (np.linalg.norm(img_pred_code-img_obs_code) ** 2) /  param.get('romi_test_size')
		print ('Current mse fwd model: ', mse)
		self.models.logger_fwd.store_log(mse)

	def run_babbling(self, param):
		p = Position()
			
		for _ in range(param.get('max_iterations')):

			# record logs and data
			self.log_current_inv_mse()
			self.log_current_fwd_mse()

			#print ('Mode ', self.goal_selection_mode, ' hist_size ', str(self.history_size), ' prob ', str(self.history_buffer_update_prob), ' iteration : ', self.iteration)
			print ('Iteration ', self.iteration)
			self.iteration = self.iteration+1

			# select a goal using the intrinsic motivation strategy
			self.current_goal_idx, self.current_goal_x, self.current_goal_y = self.intrinsic_motivation.select_goal()


			if param.get('goal_selection_mode') =='db' or param.get('goal_selection_mode') =='random' :
				self.goal_image = self.test_images[self.current_goal_idx].reshape(1, param.get('image_size'), param.get('image_size'), param.get('image_channels'))
				self.goal_code  = self.models.encoder.predict(self.goal_image)
			
			elif param.get('goal_selection_mode') =='som':
				self.goal_code  = self.models.goal_som._weights[self.current_goal_x, self.current_goal_y].reshape(1, param.get('code_size'))
			else:
				print ('wrong goal selection mode, exit!')
				sys.exit(1)

			motor_pred = []
			if param.get('goal_selection_mode') == 'db':
				motor_pred = self.models.inv_model.predict(self.goal_code)
				print ('pred ', motor_pred, ' real ', self.test_pos[self.current_goal_idx])
			else:
				goal_decoded = self.models.decoder.predict(self.goal_code)
				motor_pred = self.models.inv_model.predict(self.goal_code)
			image_pred = self.models.decoder.predict(self.models.fwd_model.predict(np.asarray(motor_pred)))

			noise_x = np.random.normal(0,0.02)
			noise_y = np.random.normal(0,0.02)
			p.x = clamp_x((motor_pred[0][0]+noise_x)*x_lims[1])
			p.y = clamp_y((motor_pred[0][1]+noise_y)*y_lims[1])

			# choose random motor commands from time to time
			ran = random.random()
			if ran < param.get('random_cmd_rate') or self.models.memory_fwd.is_memory_still_not_full() or param.get('goal_selection_mode') =='random':
				print ('generating random motor command')
				p.x = random.uniform(x_lims[0], x_lims[1])
				p.y = random.uniform(y_lims[0], y_lims[1])
				self.random_cmd_flag = True
			else:
				self.random_cmd_flag=False

			print ('predicted position ', motor_pred[0], 'p+noise ', motor_pred[0][0]+noise_x, ' ' , motor_pred[0][1]+noise_y, ' clamped ', p.x, ' ' , p.y, ' noise.x ', noise_x, ' n.y ', noise_y)

			p.z = int(-90)
			p.speed = int(1400)
		
			self.create_simulated_data(p, self.prev_pos)
			self.prev_pos=p
			'''
			if self.iteration % 50 == 0:
				#test_codes= self.encoder.predict(self.test_images[0:self.goal_size*self.goal_size].reshape(self.goal_size*self.goal_size, self.image_size, self.image_size, self.channels))
				if self.goal_selection_mode == 'db' or self.goal_selection_mode == 'random':
					plot_cvh(self.convex_hull_inv, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_inv', iteration = self.iteration, dimensions=2, log_goal=self.test_pos[0:self.goal_size*self.goal_size], num_goals=self.goal_size*self.goal_size)
				elif self.goal_selection_mode == 'kmeans':
					goals_pos = self.inverse_code_model.predict(self.kmeans.cluster_centers_[0:self.goal_size*self.goal_size].reshape(self.goal_size*self.goal_size, self.code_size))
					plot_cvh(self.convex_hull_inv, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_inv', iteration = self.iteration, dimensions=2, log_goal=goals_pos, num_goals=self.goal_size*self.goal_size)
				elif self.goal_selection_mode == 'som':
					goals_pos = self.inverse_code_model.predict(self.goal_som._weights.reshape(len(self.goal_som._weights)*len(self.goal_som._weights[0]), len(self.goal_som._weights[0][0]) ))
					plot_cvh(self.convex_hull_inv, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_inv', iteration = self.iteration, dimensions=2, log_goal=goals_pos, num_goals=self.goal_size*self.goal_size)
				#test_codes_ipca = self.fwd_ipca.fit_transform(test_codes)
				#plot_cvh(self.convex_hull_fwd, title=self.goal_selection_mode+'_'+str(self.exp_iteration)+'cvh_fwd', iteration = self.iteration, dimensions=2, log_goal=test_codes_ipca, num_goals=self.goal_size*self.goal_size)
			'''

			# update competence of the current goal (it is supposed that at this moment the action is finished
			if len(self.img)>0 and (not self.random_cmd_flag or param.get('goal_selection_mode') == 'random'):

				cmd = [p.x/float(x_lims[1]), p.y/float(y_lims[1])]
				prediction_code = self.models.fwd_model.predict(np.asarray(cmd).reshape((1,2)))

				prediction_error = np.linalg.norm(np.asarray(self.goal_code[:])-np.asarray(prediction_code[:]))
				self.intrinsic_motivation.update_competences(self.current_goal_x, self.current_goal_y, prediction_error)
				#print 'Prediction error: ', prediction_error, ' learning progress: ', self.interest_model.get_learning_progress(self.current_goal_x, self.current_goal_y)
		
				#self.log_lp.append(np.asarray(deepcopy(self.intrinsic_motivation.learning_progress)))
				#self.log_goal_id.append(self.current_goal_idx)

			# first update the memory, then update the models
			observed_pos = self.pos[-1]
			observed_img = self.img[-1]
			observed_img_code = self.models.encoder.predict(observed_img)
			self.models.memory_fwd.update(observed_pos, observed_img_code)
			self.models.memory_inv.update(observed_img_code, observed_pos)

			# fit models	
			if (len(self.img) > param.get('batch_size')) and (len(self.img) == len(self.pos)):

				observed_codes_batch = self.models.encoder.predict(self.img[-(param.get('batch_size')):]  )
				observed_pos_batch = self.pos[-(param.get('batch_size')):]

				# fit the model with the current batch of observations and the memory!
				# create then temporary input and output tensors containing batch and memory
				obs_and_mem_pos = np.vstack((np.asarray(observed_pos_batch), np.asarray(self.models.memory_fwd.input_variables)))
				obs_and_mem_img_codes = np.vstack((np.asarray(observed_codes_batch), np.asarray(self.models.memory_fwd.output_variables)))

				self.models.train_forward_code_model_on_batch(self.models.fwd_model, obs_and_mem_pos, obs_and_mem_img_codes, params)
				self.models.train_inverse_code_model_on_batch(self.models.inv_model, obs_and_mem_img_codes, obs_and_mem_pos, params)

				#train_autoencoder_on_batch(self.autoencoder, self.encoder, self.decoder, np.asarray(self.img[-32:]).reshape(32, self.image_size, self.image_size, self.channels), batch_size=self.batch_size, cae_epochs=5)

				# update convex hulls
				#obs_codes= self.encoder.predict(np.asarray(self.img[-(self.batch_size):]).reshape(self.batch_size, self.image_size, self.image_size, self.channels))

			'''
			if not self.random_cmd_flag and len(self.cmd)>0:
				#print 'test pos', self.test_positions[0:10]
				test_p = self.test_pos[self.current_goal_idx]
				#curr_cmd = self.cmd[-1]
				#pred_p = self.inverse_model.predict(self.goal_image)
				pred_p = self.inverse_code_model.predict(self.goal_code)
				self.log_goal_pos[self.current_goal_idx].append([test_p[0],test_p[1] ])
				self.log_goal_pred[self.current_goal_idx].append([pred_p[0][0], pred_p[0][1] ])
			'''
		print ('Saving models')
		self.save_models(param)

	def create_simulated_data(self, cmd, pos, param):
		self.lock.acquire()
		a = [int(pos.x), int(pos.y)]
		b = [int(cmd.x),int(cmd.y)]

		tr = self.cam_sim.get_trajectory(a,b)
		trn = self.cam_sim.get_trajectory_names(a,b)

		rounded  = self.cam_sim.round2mul(tr,5) # only images every 5mm
		for i in range(len(tr)):
			self.pos.append([float(rounded[i][0]) / x_lims[1], float(rounded[i][1]) / y_lims[1]] )
			self.cmd.append([float(int(cmd.x)) / x_lims[1], float(int(cmd.y)) / y_lims[1]] )
			cv2_img = cv2.imread(trn[i],1 )
			if param.get('channels') ==1:
				cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
			cv2_img = cv2.resize(cv2_img,(param.get('image_size'), param.get('image_size')), interpolation = cv2.INTER_LINEAR)
			cv2_img = cv2_img.astype('float32') / 255
			cv2_img.reshape(1, param.get('image_size'), param.get('image_size'), param.get('image_channels'))
			self.img.append(cv2_img)

		self.lock.release()


	def save_models(self, param):
		self.models.save_models(param)

		print ('Models saved')
		self.goto_starting_pos()

	def clear_session(self):
		# reset
		print('Clearing TF session')
		if tf.__version__ < "1.8.0":
			tf.reset_default_graph()
		else:
			tf.compat.v1.reset_default_graph()

	def Exit_call(self, signal, frame):
		self.goto_starting_pos()
		self.save_models()

	def get_starting_pos(self):
		p = Position()
		p.x = int(0)
		p.y = int(0)
		p.z = int(-50)
		p.speed = int(1400)
		return p

	def goto_starting_pos(self):
		p = self.get_starting_pos()
		self.create_simulated_data(p, self.prev_pos)
		self.prev_pos=p


if __name__ == '__main__':

	# reset
	print('Clearing TF session')
	if tf.__version__ < "1.8.0":
		tf.reset_default_graph()
	else:
		tf.compat.v1.reset_default_graph()

	goal_babbling = GoalBabbling()

	parameters = Parameters()
	parameters.set('goal_selection_mode', 'som')
	parameters.set('exp_iteration', 0)
	print ('Starting experiment')
	goal_babbling.initialise(parameters)
	goal_babbling.run_babbling()
	print ('Experiment done')

	'''
	os.chdir('experiments')
	exp_iteration_size = 5
	exp_type = ['db', 'som', 'random']#, 'kmeans']
	history_size = [0, 10, 20]
	prob = [0.1, 0.01]

	for e in range(len(exp_type)):
		print ('exp ', exp_type[e])

		for h in range( len (history_size)):
			print('history size ', history_size[h])

			for p in range(len(prob)):
				print('prob update ', prob[p])

				for i in range(exp_iteration_size):
					print( 'exp ', exp_type[e], ' history size ', str(history_size[h]), ' prob ', str(prob[p]), ' iteration ', str(i) )
					directory = './'+exp_type[e]+'_'+str(history_size[h])+'_'+str(prob[p])+'_'+str(i)+'/'
					if not os.path.exists(directory):
						os.makedirs(directory)

						if not os.path.exists(directory+'models'):
							os.makedirs(directory+'models')

						shutil.copy('../pretrained_models/autoencoder.h5', directory+'models/autoencoder.h5')
						shutil.copy('../pretrained_models/encoder.h5', directory+'models/encoder.h5')
						shutil.copy('../pretrained_models/decoder.h5', directory+'models/decoder.h5')
						shutil.copy('../pretrained_models/goal_som.h5', directory+'models/goal_som.h5')
						shutil.copy('../pretrained_models/kmeans.sav', directory+'models/kmeans.sav')

						os.chdir(directory)
						if not os.path.exists('./models/plots'):
							os.makedirs('./models/plots')
						if not os.path.exists('./data'):
							os.makedirs('./data')
						print ('current directory: ', os.getcwd())

						goal_babbling.initialise( goal_selection_mode= exp_type[e], exp_iteration = i, hist_size= history_size[h], prob_update=prob[p])
						goal_babbling.run_babbling()
						os.chdir('../')
						#GoalBabbling().
						print ('finished experiment ', exp_type[e], ' history size ', str(history_size[h]),' prob ', str(prob[p]), ' iter ', str(i))

						goal_babbling.clear_session()
					print ('experiment ', directory, ' already carried out')
	os.chdir('../')
	plot_learning_comparisons(model_type = 'fwd', exp_size = exp_iteration_size, save = True, show = True)
	plot_learning_comparisons(model_type = 'inv', exp_size = exp_iteration_size, save = True, show = True)
	'''
