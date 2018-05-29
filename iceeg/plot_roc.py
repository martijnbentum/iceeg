import glob
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import path
import time

class cm_collection:
	'''A class to collect all confusion_matrix objects and plot them.'''
	def __init__(self,cm = [],load_all = True):
		if cm != []: load_all = False
		self.cm = cm
		self.perc2color = {'50':'blue','10':'green','25':'orange','40':'purple','5':'cyan'}
		if load_all: self.load_all_cm()
		self.set_perc_info()



	def set_perc_info(self):	
		'''Find all perc artifacts present in the collection of cm objects.'''
		self.color_patch,self.percs = [],[]
		for cm in self.cm:
			if cm.perc not in self.percs: self.percs.append(cm.perc)

		self.percs = list(map(int,self.percs))
		self.percs.sort()
		self.percs = list(map(str,self.percs))	


		for perc in self.percs:
			self.color_patch.append( mpatches.Patch(color=self.perc2color[perc], label=perc + '%') )

	def load_all_cm(self):
		'''Load all cm objects.'''
		self.cm = []
		self.fn = get_channel_cm_fn()
		for f in self.fn:
			self.cm.append(confusion_matrix(f))
	
	def cluster_perc(self):
		'''NOT USED, cluster cm object on perc.'''
		self.cm_perc = {}
		for perc in self.percs:
			self.cm_perc[perc] = []

		for cm in self.cm:
			for perc in self.percs:
				if perc == cm.perc:
					self.cm_perc[perc].append(cm)

	def find_last(self):
		'''Find cm that was created last.'''
		self.current_time = time.time()
		self.last_cm = self.cm[0]
		dif = self.current_time - self.last_cm.created_epoch
		for cm in self.cm:
			new_dif = self.current_time -cm.created_epoch 
			if new_dif < dif:
				self.last_cm = cm
				dif = new_dif
		
	def color_perc(self):
		'''Color each cm based on perc artifact value (percentage of artifacts in training.'''
		for cm in self.cm:
			if cm.perc in self.perc2color.keys():
				cm.color = self.perc2color[cm.perc]
			else: cm.color = 'grey'

	def plot(self, plot_type = 'roc', save = False):
		'''Plot ROC / Precision-Recall / MCC for all cm objects.'''
		self.find_last()
		self.color_perc()
		self.figure = plt.figure()
		for cm in self.cm:
			if cm == self.last_cm: 
				alpha = 1
				cm.marker = '*'
				markersize =24 
			else: 
				alpha = float(cm.part)/100
				markersize =6 
			if plot_type == 'roc':
				cm.plot_roc(self.figure,alpha= alpha, markersize = markersize)
			if plot_type == 'mcc': 
				cm.plot_mcc(self.figure,alpha= alpha, markersize = markersize)
			if plot_type == 'pr':
				cm.plot_pr(self.figure,alpha= alpha, markersize = markersize)
			if plot_type == 'f':
				cm.plot_f(self.figure,alpha= alpha, markersize = markersize)
		plt.grid()
		if plot_type != 'f':plt.legend(handles = self.color_patch)
		else: plt.legend(('f1','precision','recall'))
		if plot_type == 'roc':
			plt.xlim((0,0.3))
			plt.ylim((0,1))
			plt.xlabel('false positive rate')
			plt.ylabel('recall')
			plt.title('ROC curve')
		if plot_type == 'mcc':
			plt.ylabel('mcc')
			plt.xlabel('n parts training')
			plt.title('Matthews correlation coefficient')
		if plot_type == 'pr':
			plt.xlim((0,1))
			plt.ylim((0,1))
			plt.ylabel('precision')
			plt.xlabel('recall')
			plt.title('Precision / Recall')
		if plot_type == 'f':
			plt.ylabel('score')
			plt.xlabel('part')
			plt.title('F1 - Precision - Recall')
			
	
		if save:
			plt.savefig(plot_type+'.png')
			plt.close()
			
	

class confusion_matrix:
	'''An object to represent confusion matrix, especially created with respect to model training.
	'''
	def __init__(self,filename = '',data = '',part = '',fold ='', kernel = '',perc = '',model= '',tp = '',rep = 0,color = 'grey', marker = 'o'):
		'''Create cm object based on filename of numpy stored confusion matrix or by 
		providing numpy array containing confusion matrix

		filename 		filename of numpy array containing confusion matrix
		data 			numpy array containing confusion matrix
		part 			the training part corresponding with cm
		fold 			fold of the data object used in model training
		kernel 			kernel used in model training
		model 			model file used in model training
		tp  			test part used to evaluate model
		color 			color of the cm if plotted
		marker 			marker shape if plotted
		'''

		self.cm = data
		self.part = part
		self.fold = fold
		self.kernel = kernel
		self.perc = perc
		self.model = model
		self.tp = tp
		self.rep = rep
		self.cm_filename = filename
		self.color = color
		self.marker = marker
		if filename:
			self.load_data()
			self.filename2info()
			self.load_report()
			self.created = time.ctime(os.path.getctime(self.cm_filename))
			self.created_epoch = os.path.getctime(self.cm_filename)
		self.precision_recall()


	def __str__(self):
		m = 'part\t' + self.part
		m += 'fold\t' + self.fold
		m += 'kernel\t' + self.kernel
		m += 'perc\t' + self.perc
		m += 'model\t' + self.model
		m += 'color\t' + self.color
		m += 'marker\t' + self.marker
		m += 'fpr\t' + str(self.fpr)
		m += 'recall\t' + str(self.recall)
		m += 'precision\t' + str(self.precision)
		m += 'confusion matrix\n' + np.array2string(self.cm)
		m += '\n\n' + self.report
		
	def __repr__(self):
		return 'cm object\tperc: ' + self.perc +'\tpart: ' + self.part +'\tkernel: '+self.kernel +'\tmcc: ' + str(round(self.mcc,2))

	def load_data(self):
		self.cm = np.load(self.cm_filename)

	def load_report(self):
		'''Load the sklearn report generated by model_ch_cnn corresponding with cm.'''
		self.report_filename = self.cm_filename.replace('cm.npy','report.txt')
		self.report = open(self.report_filename).read()


	def filename2info(self):
		'''Set available info fields with correct info if present (based on filename).'''
		items = self.cm_filename.split('_')
		for item in items:
			if 'perc' in item: self.perc = item.split('-')[-1]
			if 'part' in item: self.part = item.split('-')[-1]
			if 'fold' in item: self.fold = item.split('-')[-1]
			if 'tp' in item: self.tp= item.split('-')[-1]
			if 'kernel' in item: self.kernel= item.split('model')[0].split('-')[-1]
			if 'model' in item: self.model= item.split('model')[-1]
			if 'rep' in item: self.rep = int(item.split('-')[-1]) -1


	def plot_roc(self,figure = None, alpha = 1,markersize = 10):
		'''Plot recall/fpr value of cm.'''
		if figure == None:
			if not hasattr(self,'figure'): self.figure = plt.figure()
		else: self.figure = figure
		# plt.figure(self.figure.number)
		plt.plot(self.fpr,self.recall,color = self.color, alpha = alpha, marker = self.marker,markersize = markersize)
		
	def plot_mcc(self,figure = None, alpha = 1,markersize = 10):
		'''Plot matthews correlation coefficient.'''
		if figure == None:
			if not hasattr(self,'figure'): self.figure = plt.figure()
		else: self.figure = figure
		# plt.figure(self.figure.number)
		plt.plot(int(self.part)+int(self.rep)*90,self.mcc,color = self.color, alpha = alpha, marker = self.marker,markersize = markersize)

	def plot_pr(self,figure = None, alpha = 1,markersize = 10):
		'''Plot precision/recall value of cm.'''
		if figure == None:
			if not hasattr(self,'figure'): self.figure = plt.figure()
		else: self.figure = figure
		# plt.figure(self.figure.number)
		plt.plot(self.recall,self.precision,color = self.color, alpha = alpha, marker = self.marker,markersize = markersize)

	def plot_f(self,figure = None, alpha =1, markersize =10):
		'''Plot matthews correlation coefficient.'''
		if figure == None:
			if not hasattr(self,'figure'): self.figure = plt.figure()
		else: self.figure = figure
		# plt.figure(self.figure.number)
		plt.plot(int(self.part)+int(self.rep)*90,self.f1,color = 'purple', alpha = alpha, marker = self.marker,markersize = markersize)
		plt.plot(int(self.part)+int(self.rep)*90,self.precision,color = 'red', alpha = alpha, marker = self.marker,markersize = markersize)
		plt.plot(int(self.part)+int(self.rep)*90,self.recall,color = 'blue', alpha = alpha, marker = self.marker,markersize = markersize)
		
	
	def precision_recall(self):
		'''Compute precision, recall, f1, mmc and false positive rate based on the confusion matrix.'''
		cm = self.cm
		tn, fp, fn, tp = cm.ravel()
		self.tn, self.fp, self.fn, self.tp = cm.ravel()
		self.fpr = fp / (fp+tn)
		self.recall = tp / (tp + fn)
		self.precision = tp / (tp+fp)
		self.f1 = 2 * (1 / ( (1/self.recall) + (1/self.precision) ))
		self.mcc = matthews_correlation_coefficient(cm)


def get_channel_cm_fn():
	'''Get all cm filenames in the model_channel directory.'''
	return glob.glob(path.model_channel + '*cm.npy')


def find_scale(cm):
	'''calculate a scale factor to prevent overflow during calculation of the mcc.'''
	smallest = np.min(cm)
	scale = 1
	while 1:
		if smallest%scale == smallest:
			if scale == 1: return scale
			return int(scale/10)
		scale *= 10


def matthews_correlation_coefficient(cm):
	'''Calculate the mcc based on the confusion matrix.
	-1 perfect disagreement of predicted and ground truth, 0 random agrement, 1 perfect agreement.
	'''
	# print(cm.dtype)
	if cm.shape != (2,2): 
		print('confusion matrix should be 2X2 is:',cm.shape)
		return 0
	cm = cm.astype(np.int64)
	scale = find_scale(cm)
	cm = cm / scale
	tn, fp, fn, tp = cm.ravel()
	numerator = (tp*tn - fp*fn) 
	# if any of the sums in the denominator == 0, set arbitrarily to 1, this will result in a mcc of 0
	if tp+fp == 0 or tp+fn == 0 or tn+fp == 0 or tn+fn ==0: denominator = 1
	else:denominator = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) ** 0.5
	# print(numerator, denominator,tp,tn,fp,fn)
	return numerator/denominator
