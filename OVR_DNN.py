import numpy as np
import pandas as pd
import pickle
from skmultilearn.model_selection import iterative_train_test_split
from imblearn.over_sampling import RandomOverSampler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier

import glob

class OVR_DNN:
    
    def __init__(self, X_train=None, y_train=None, filename=None):
        self._X_train = X_train
        self._y_train = y_train
        self._filename = filename
        if self._X_train is not None:
            self.train_base_models()
            self.train_stk_models()
        if self._filename is not None:
            print('Loading models...')
            self.load_models()
        
        
        
    def train_base_models(self):
        self.split_data()
        estimators = self.get_models()
        first_one = True
        feat_imp_sums = np.zeros(self._X_train.shape[1])
        for pair in estimators:
            print('Training base model', pair[0])
            pair[1].fit(self._X_train, self._y_train)
            for est in pair[1].estimators_:
                if hasattr(est, 'feature_importances_'):
                    feat_imp_sums += est.feature_importances_
        self._base_models = estimators
        self._imp_feats = feat_imp_sums > np.quantile(feat_imp_sums, 0.8)
        
        
        
        
    def get_models(self):
        base_lr = LogisticRegression(class_weight='balanced')
        ovr_lr = OneVsRestClassifier(base_lr)

        base_eec = EasyEnsembleClassifier(n_estimators=10)
        ovr_eec = OneVsRestClassifier(base_eec)

        base_rus = RUSBoostClassifier(n_estimators=50)
        ovr_rus = OneVsRestClassifier(base_rus)

        base_bbc = BalancedBaggingClassifier(n_estimators=10)
        ovr_bbc = OneVsRestClassifier(base_bbc)

        base_brf = BalancedRandomForestClassifier(n_estimators=100)
        ovr_brf = OneVsRestClassifier(base_brf)

        estimators = [('lr', ovr_lr), ('eec', ovr_eec),
                      ('bbc', ovr_bbc),
                      ('brf', ovr_brf)]
        return estimators
    
    
    def split_data(self):
        print('Splitting data into training and validation set to train DNNs...')
        X_train, y_train, X_val, y_val = iterative_train_test_split(self._X_train, 
                                                              self._y_train, 
                                                              test_size = 0.35)
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        print('Train data:', X_train.shape)
        print('Train labels:', y_train.shape)
        print('Val data:', X_val.shape)
        print('Val labels:', y_val.shape)
            
            
    def train_stk_models(self):
        print('Training stacking model on validation set...')
        for i,model in enumerate(self._base_models):
            print('  Getting probabilities for validation set...')
            this_y_prob = model[1].predict_proba(self._X_val)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)
                
        stk_brf = BalancedRandomForestClassifier(n_estimators=400)
        ovr_stk_brf = OneVsRestClassifier(stk_brf)
        ovr_stk_brf.fit(y_prob, self._y_val)
        self._stk_model = ovr_stk_brf
    
    
    
    def predict(self, X_test):
        for i,model in enumerate(self._base_models):
            this_y_prob = model[1].predict_proba(X_test)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)
        
        y_pred_test = self._stk_model.predict(y_prob)
        return y_pred_test
    
    
    
    def predict_proba(self, X_test):
        for i,model in enumerate(self._base_models):
            this_y_prob = model[1].predict_proba(X_test)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)
        
        y_pred_test = self._stk_model.predict_proba(y_prob)
        return y_pred_test
    
    
    
    def save_models(self, filename):
        model_dict = {
            'base_models': self._base_models,
            'imp_feats': self._imp_feats,
            'stk_model': self._stk_model
        }
        stk_filename = filename.split('.pick')[0] + '_ovr_imb_models.pickle'
        with open(stk_filename, 'wb') as handle:
            pickle.dump(model_dict, handle)
        handle.close()
        
    
    
    
    def load_models(self):
        # load in base dnn models
        file_hash = self._filename.split('.pick')[0] + '_ovr_imb_models.pickle'
        with open(file_hash, 'rb') as handle:
            model_dict = pickle.load(handle)
        handle.close()
        self._base_models = model_dict['base_models']
        self._imp_feats = model_dict['imp_feats']
        self._stk_model = model_dict['stk_model']
    
    
    