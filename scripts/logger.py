import numpy as np
from copy import deepcopy

class Logger:

    def __init__(self, param):
        self.parameters = param
        # contains the list of MSE calculation over time
        self.mse = []

        self.count_of_changed_memory_elements = []
        
        self.input_variances= []
        self.output_variances= []
        self.learning_progress = []

    def store_log(self, mse = [], input_var = [], output_var = [], count_of_changed_memory_elements = [], learning_progress = []):
        self.mse.append(mse)
        self.input_variances.append(input_var)
        self.output_variances.append(output_var)
        self.count_of_changed_memory_elements.append(count_of_changed_memory_elements)
        self.learning_progress.append(deepcopy(learning_progress))
        #print (str(self.learning_progress))

    def store_log(self, mse = []):
        self.mse.append(mse)

    def get_iteration_count(self):
        return len(self.mse)