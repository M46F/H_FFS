from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import f_classif

from .fcbf import FCBFK

import numpy as np

import pymrmr
import time
import copy


def bcr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 0.5 * ((tp/(tp+fn)) + (tn/(tn+fp)))


class FFS():

	def __init__(self, make_clf, clf_param={}, K=10, scorer=bcr_score):
		self.K = K
		self.make_clf = make_clf
		self.clf_param = clf_param
		self.scorer = scorer

	def fit(self, X_train, y_train, X_val=None, y_val=None):
		if X_val is None or y_val is None:
			X_val = X_train
			y_val = y_train
		self.ffs(X_train, y_train, X_val, y_val)

	def transform(self, X):
		if not self.isFitted or self.k == 0:
			raise ValueError('not yet fitted')
		return X[X.columns[self.results_[self.k]['selected_features']]]

	def fit_transform(self, X_train, y_train, X_val=None, y_val=None):
		self.fit(X_train, y_train, X_val, y_val)
		return self.transform(X_train)

	def ffs(self, X_train, y_train, X_val, y_val, K=None):
		if K is None:
			K = self.K
		selected = set()
		self.results_ = {}
		self.k = 0
		all_features = set(range(len(X_train.columns)))
		if K > len(all_features):
		    raise ValueError('K cannot be higher than number of features')
		while self.k < K:
		    start_time = time.time()
		    max_score = -1
		    choosen = None
		    for feat in all_features:
		        current_features = list(selected.union({feat}))
		        current_features.sort()
		        clf = self.make_clf(**self.clf_param)
		        clf.fit(X_train[X_train.columns[current_features]], y_train)
		        current_score = self.scorer(y_true=y_val, y_pred=clf.predict(X_val[X_val.columns[current_features]]))
		        if current_score > max_score:
		            choosen = feat
		            max_score = current_score
		    selected.add(choosen)
		    all_features.remove(choosen)
		    self.k+=1
		    end_time = time.time()
		    self.results_[self.k] = {
		        'score': max_score,
		        'time': end_time - start_time,
		        'total_time': end_time - start_time if self.k==1 else self.results_[self.k-1]['total_time'] + (end_time - start_time),
		        'selected_features': copy.copy(selected),
		        'k':copy.copy(self.k)
		    }


def get_anova(X_train, y_train, features, C, **params):
	list_features = np.array(list(features))
	anova = f_classif(X_train[X_train.columns[list_features]], y_train)
	arg_sorted = np.argsort(anova[0])
	return list_features[arg_sorted[0:C]]


def get_mrmr(X_train, _, features, C, **params):
	list_features = np.array(list(features))
	selected_X = X_train[X_train.columns[list_features]]
	selected_col = pymrmr.mRMR(selected_X, 'MIQ', C)
	return np.array([X_train.columns.get_loc(c) for c in selected_col])


def get_fcbf(X_train, y_train, features, C, **params):
	list_features = np.array(list(features))
	fs = FCBFK(k=C)
	fs.fit(X_train[X_train.columns[list_features]].values,y_train.values)
	return list_features[fs.idx_sel]


class H_FFS(FFS):
	def __init__(self, 
		make_clf, 
		clf_param={},
		K=10,
		C=10,
		scorer=bcr_score, 
		method='ANOVA',
		hybrid_func=None, 
		hybrid_param={},
		memorization = False
		):
		self.K = K
		self.C = C
		self.make_clf = make_clf
		self.clf_param = clf_param
		self.scorer = scorer
		if method in ['ANOVA', 'MRMR', 'FCBF'] and hybrid_func is None:
			if method == 'ANOVA':
				self.hybrid_func = get_anova
				self.hybrid_param = {}
			elif method == 'MRMR':
				self.hybrid_func = get_mrmr
				self.hybrid_param = {}
			else:
				self.hybrid_func = get_fcbf
				self.hybrid_param = {}
		else:
			self.hybrid_func = hybrid_func
			self.hybrid_param = hybrid_param	

	def ffs(self, X_train, y_train, X_val, y_val, K=None):
		if K is None:
			K = self.K
		selected = set()
		self.results_ = {}
		self.k = 0
		all_features = set(range(len(X_train.columns)))
		if K > len(all_features):
		    raise ValueError('K cannot be higher than number of features')
		while self.k < K:
		    start_time = time.time()
		    max_score = -1
		    choosen = None
		    heuristic_features = self.hybrid_func(X_train, y_train, all_features, self.C, **self.hybrid_param)
		    for feat in heuristic_features:
		        current_features = list(selected.union({feat}))
		        current_features.sort()
		        clf = self.make_clf(**self.clf_param)
		        clf.fit(X_train[X_train.columns[current_features]], y_train)
		        current_score = self.scorer(y_true=y_val, y_pred=clf.predict(X_val[X_val.columns[current_features]]))
		        if current_score > max_score:
		            choosen = feat
		            max_score = current_score
		    selected.add(choosen)
		    all_features.remove(choosen)
		    self.k+=1
		    end_time = time.time()
		    self.results_[self.k] = {
		        'score': max_score,
		        'time': end_time - start_time,
		        'total_time': end_time - start_time if self.k==1 else self.results_[self.k-1]['total_time'] + (end_time - start_time),
		        'selected_features': copy.copy(selected),
		        'k':copy.copy(self.k)
		    }
