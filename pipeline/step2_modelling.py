import numpy as np
import yaml

from logic.modelling import (evaluate_model_with_cv,
                             train_model)
from utils import save_data, load_data


class Modeller():
    def __init__(self, config_path='config/step2_modelling.yaml'): 
        # General attributes
        with open(config_path) as stream:
            self.config = yaml.safe_load(stream)
        
        self.classifier_data = \
            load_data(self.config['input']['classifier_data_path'])
        self.classifier_cv_score = None
        self.classifier_model = None

    def run(self):
        # Split data into predictors and targets
        X, y = self._split_data()
        self.classifier_cv_score = evaluate_model_with_cv(X, y, self.config)
        self.classifier_model = train_model(X, y)
        save_data(self.classifier_model, self.config['output']['model_path'])

    def _split_data(self):
        X = np.asarray(self.classifier_data[:,:-1])
        y = np.asarray(self.classifier_data[:,-1])
        return X, y