from __future__ import print_function # added this to support usage of python2

import os
import numpy as np
from enum import Enum
import pickle

class MemUpdateStrategy(Enum):
    HIGH_LEARNING_PROGRESS = 0
    LOW_LEARNING_PROGRESS = 1
    RANDOM = 2


class Parameters:

    def __init__(self):
        self.dictionary = {
            'directory': '',
            'image_size': 64,
            'image_channels': 1,
            'code_size': 32,
            'goal_size': 3,

            'batch_size':16,
            'epochs': 1, # online fwd/inv models learning
            'goal_selection_mode':'som',
            'exp_iteration': 0,

            'max_iterations':5000,

            'cae_filename': 'autoencoder.h5',
            'encoder_filename': 'encoder.h5',
            'decoder_filename':'decoder.h5',
            'cae_batch_size':32,
            'cae_epochs':2,
            'cae_max_pool_size': 2,
            'cae_conv_size': 3,

            'romi_input_dim': 2,
            'romi_dataset_pkl': 'romi_data/compressed_dataset.pkl',
            'romi_test_data_step': 500,
            'romi_test_size': 50,

            'fwd_filename': 'forward_code_model.h5',
            'inv_filename': 'inverse_code_model.h5',
            'som_filename': 'goal_som.h5',


            'random_cmd_flag': False,
            'random_cmd_rate': 0.3,

            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'memory_size': 1000,
            'memory_update_probability': 0.0001,
            'memory_update_strategy': MemUpdateStrategy.RANDOM,  # possible choices:  random, learning_progress
            #'batch_size': 32,
            'batchs_to_update_online': 3,
            'mse_test_dataset_fraction' : 20,  #   how many samples to use in the MSE calculations? dataset_size / this.
            'mse_calculation_step': 4, # calculate MSE every X model fits
            'experiment_repetition': -1,
            'verbosity_level': 1
        }

    def get(self, key_name):
        if key_name in self.dictionary.keys():
            return self.dictionary[key_name]
        else:
            print('Trying to access parameters key: '+ key_name+ ' which does not exist')

    def set(self, key_name, key_value):
        if key_name in self.dictionary.keys():
            print('Setting parameters key: ', key_name, ' to ', str(key_value))
            self.dictionary[key_name] = key_value
        else:
            print('Trying to modify parameters key: '+ key_name+ ' which does not exist')

    def save(self):
        pickle.dump(self.dictionary, open(os.path.join(self.get('directory'), 'parameters.pkl'), 'wb'),  protocol=2) # protcolo2 for compatibility with python2
        # save also as plain text file
        with open(os.path.join(self.get('directory'), 'parameters.txt'), 'w') as f:
            print(self.dictionary, file=f)
