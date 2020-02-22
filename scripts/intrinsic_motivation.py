# author Guido Schillaci, Dr.rer.nat. - Scuola Superiore Sant'Anna
# Guido Schillaci <guido.schillaci@santannapisa.it>
import numpy as np
import random
import sys
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import os
import utils

class IntrinsicMotivation():

	def __init__(self, param):
		self.param = param
		#self.learning_progress = 0.0 # interest factor for each goal

		self.pe_buffer = [] # buffer of prediction errors for each goal. Its size is dynamic!
		for i in range(self.param.get('goal_size')*self.param.get('goal_size')):
			self.pe_buffer.append([])
		self.pe_max_buffer_size_history = [] # keep track of the buffer size over time (same size for all goals)

		self.slopes_pe_dynamics = [] # for each goal, keeps track of the trends of the PE_dynamics (slope of the regression over the pe_buffer)

		self.goal_id_history = [] # keep track of the goals that have been selected over time

		self.movements_amplitude = []

		# self.pe_derivatives = [] # derivative of the prediction errors for each goal
		#self.competence_measure = param.get('im_competence_measure')
		#self.decay_factor = param.get('im_decay_factor')

		if self.param.get('goal_size') <=0:
			print ("Error: goal_size <=0")
			sys.exit(1)

		#self.learning_progress= np.resize(self.learning_progress, (self.param.get('goal_size')*self.param.get('goal_size') ,))
		#for i in range (0, self.param.get('goal_size')*self.param.get('goal_size')):
	#		self.pe_history.append([0.0])
			#self.pe_derivatives.append([0.0])
		print ("Intrinsic motivation initialised. PE buffer size: ", len(self.pe_buffer))

	def select_goal(self):
		ran = random.random()
		goal_idx = 0
		if ran < self.param.get('im_random_goal_prob') or len(self.slopes_pe_dynamics)==0:
			# select random goal
			goal_idx = random.randint(0, self.param.get('goal_size') * self.param.get('goal_size') - 1)
			print ('Interest Model: Selected random goal. Id: ', goal_idx)
		else:
		#if ran < 0.85 and np.sum(self.learning_progress)>0: # 70% of the times
			# select best goal
			goal_idx = self.get_best_goal_index()
			print ('Intrinsic motivation: Selected goal id: ', goal_idx)

		goal_x =int( goal_idx / self.param.get('goal_size')  )
		goal_y = goal_idx % self.param.get('goal_size')
		print ('Goal_SOM coords: ', goal_idx, ' x: ', goal_x, ' y: ', goal_y)
		return goal_idx, goal_x, goal_y

	def update_max_pe_buffer_size(self):
		if self.param.get('im_fixed_pe_buffer_size'):
			self.pe_max_buffer_size_history.append(self.param.get('im_initial_pe_buffer_size'))
		else:
			new_buffer_size = 0.0 # TODO: make it dependent on the overall PE reduction trend
			self.pe_max_buffer_size_history.append(new_buffer_size)

	def get_goal_id(self, id_x, id_y):
		goal_id = int(id_x * self.param.get('goal_size') + id_y)
		if (goal_id <0 or goal_id>(self.param.get('goal_size')*self.param.get('goal_size'))):
			print ("Intrinsic motivation error, wrong goal id: ", goal_id)
			sys.exit(1)
		return goal_id


	def update_error_dynamics(self, goal_id_x, goal_id_y, prediction_error):
		print ('updating error dynamics')
		if len(self.pe_max_buffer_size_history)==0:
			self.pe_max_buffer_size_history.append(self.param.get('im_initial_pe_buffer_size'))
		else:
			self.update_max_pe_buffer_size()

		goal_id = self.get_goal_id(goal_id_x, goal_id_y)
		# keep track of the goal that have been selected
		self.goal_id_history.append(goal_id)

		# append the current predction error to the current goal
		self.pe_buffer[goal_id].append(prediction_error)

		# for each goal
		# - check that all the buffers are within the max buffer size
		# - compute regression on prediction error and save the slope (trend of error dynamics)
		current_slopes_err_dynamics = []
		for i in range(self.param.get('goal_size')*self.param.get('goal_size')):
			while len(self.pe_buffer[i]) > self.pe_max_buffer_size_history[-1]:
				self.pe_buffer[i].pop(0) #remove first element
			if len(self.pe_buffer[i]) < 2 : # not enough prediction error to calculate the regression
				current_slopes_err_dynamics.append(0)
			else:
				# get the slopes of the prediction error dynamics
				regr_x = np.asarray(range(len(self.pe_buffer[i]))).reshape((-1,1))
				print ('calculating regression on goal ', str(i))
				model = LinearRegression().fit(regr_x, np.asarray(self.pe_buffer[i]))
				current_slopes_err_dynamics.append(model.coef_[0]) # add the slope of the regression

		self.slopes_pe_dynamics.append(current_slopes_err_dynamics)
		print ('slopes', self.slopes_pe_dynamics[-1])
		


	# get the index of the goal associated with the lowest slope in the prediction error dynamics
	def get_best_goal_index(self):
		return np.argmin(self.slopes_pe_dynamics[-1])

	def log_last_movement(self, pos_a, pos_b):
		self.movements_amplitude.append(utils.distance(pos_a,pos_b))

	def get_linear_correlation_btw_amplitude_and_pe_dynamics(self):
		# first make a vector storing the pe_dynamics of the current goals over time
		self.slopes_of_goals = [[ self.slopes_pe_dynamics[elem][self.goal_id_history[elem]] ] for elem in self.goal_id_history]
		self.pearson_corr = pearsonr(self.slopes_of_goals, self.movements_amplitude)
		print ('Pearson correlation', self.pearson_corr)
		self.plot_slopes_of_goals()
		return self.pearson_corr


	def save_im(self):
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_pe_dynamics'), self.slopes_pe_dynamics)
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_goals'), self.slopes_of_goals)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pe_max_buffer_size_history'), self.pe_max_buffer_size_history)
		np.save(os.path.join(self.param.get('results_directory'), 'im_goal_id_history'), self.goal_id_history)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr'), self.pearson_corr)

	def plot_slopes(self, save=True, show=True):
		fig = plt.figure(figsize=(10, 10))
		num_goals = self.param.get('goal_size') * self.param.get('goal_size')
		ax1 = plt.subplot(num_goals + 1, 1, 1)
		plt.plot(self.goal_id_history)
		plt.ylim(1, num_goals)
		plt.ylabel('goal id')
		plt.xlabel('time')
		ax1.set_yticks(np.arange(0, num_goals))
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		# data = np.transpose(log_lp)
		data = np.transpose(self.slopes_pe_dynamics)
		for i in range(0, num_goals):
			ax = plt.subplot(num_goals + 1, 1, i + 2)
			plt.plot(data[i])
			plt.ylabel('g{}'.format(i))

		if save:
			plt.savefig(self.param.get('results_directory') + '/plots/im_slopes_pe_dynamics.jpg')
		if show:
			plt.show()
		plt.close()

	def plot_slopes_of_goals(self, save=True, show=True):
		fig = plt.figure(figsize=(10, 10))
		num_goals= self.param.get('goal_size')*self.param.get('goal_size')
		ax1 = plt.subplot(num_goals + 1, 1, 1)
		plt.plot(self.slopes_of_goals)
		plt.ylabel('slope')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		if save:
			plt.savefig(self.param.get('results_directory')+'/plots/im_slopes_of_goals.jpg')
		if show:
			plt.show()
		plt.close()
