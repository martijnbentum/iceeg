import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import path
import scipy.signal as signal
import sklearn.metrics

'''Functions for testing the thresholding approch of artifact detection.'''


def hamming_data(d):
	'''Multiply each window with a a hamming window.'''
	window_length = d.shape[1]
	hamming = signal.hamming(window_length)
	return d * hamming


def find_threshold_artifacts_indices(d,threshold):
	'''Find all indices of epochs with values that exceed +- threshold.'''
	lower_threshold = threshold * -1
	higher_threshold = threshold 

	to_low = np.where(np.min(d,axis=1) < lower_threshold)[0]
	to_heigh = np.where(np.max(d,axis=1) > higher_threshold)[0]
	artifact_indices = list(set(list(to_heigh) + list(to_low)))
	return artifact_indices

def load_data(f):
	'''Load a numpy array.'''
	return np.load(f)

def save_indices(name,indices):
	'''Save a numpy array containing indices.'''
	np.save(name,indices)

def make_indices_output_filename(original_name, threshold):
	'''Create a filename for an indices array.'''
	return path.snippet_annotation + original_name.split('/')[-1].split('.')[0] + '_T-' + str(threshold) + '_indices'

def fni2fnd(f,fn_data):
	'''Convert an indices filename to a datafilename.'''
	fid = f.split('/')[-1].split('.')[0]
	for fd in fn_data:
		if fid in fd:
			return fd
	# raise ValueError('corresponding filename not in fn_data',f)
	return False

def make_threshold_indices(d,threshold = 75):
	'''Create an array (nrows_d X 1) with 1 at index that is an artifact according to the provided threshold, zero otherwise.'''
	artifact_indices = find_threshold_artifacts_indices(d,threshold)
	threshold_indices = np.zeros(d.shape[0])
	threshold_indices[artifact_indices] = 1
	return threshold_indices

def compare_artifacts_groundtruth(gt_indices,threshold_indices):
	'''Evaluate threshold indices on the manually coded data.'''
	confusion_matrix = sklearn.metrics.confusion_matrix(gt_indices,threshold_indices)
	report = sklearn.metrics.classification_report(gt_indices,threshold_indices)
	mcc = matthews_correlation_coefficient(confusion_matrix)
	print(confusion_matrix)
	print('Matthews correlation coefficient:',mcc)
	if mcc > .3: print(report)
	return confusion_matrix,report,mcc


def precision_recall(cm):
	'''Compute precision, recall, f1, mmc and false positive rate based on the confusion matrix.'''
	recall_0 = cm[0,0] / (cm[0,0] + cm[0,1])
	precision_0 = cm[0,0] / (cm[0,0] + cm[1,0])
	f1_0 = 2 * (1 / ( (1/recall_0) + (1/precision_0) ))

	recall_1 = cm[1,1] / (cm[1,1] + cm[1,0])
	precision_1 = cm[1,1] / (cm[1,1] + cm[0,1])
	f1_1 = 2 * (1 / ( (1/recall_1) + (1/precision_1) ))
	mcc = matthews_correlation_coefficient(cm)

	fpr_0 = cm[1,0]/ (cm[1,0] + cm[1,1])
	fpr_1 = cm[0,1]/ (cm[0,1] + cm[0,0])
	return recall_0,precision_0,f1_0,recall_1,precision_1,f1_1,mcc,fpr_0,fpr_1

def make_precision_recall_dict(output_dict):
	'''Create dictionary with all thresholds linked to corresponding confusion matrix / precision and recall information.'''
	pr_dict,cm_dict = {},{}
	for k in output_dict:
		for threshold in range(40,201,10):
			if output_dict[k][0].shape == (2,2) and k[1] == threshold: 
				if threshold in cm_dict.keys(): cm_dict[threshold] += output_dict[k][0]
				else: cm_dict[threshold] = output_dict[k][0]
	for k in cm_dict:
		pr_dict[k] = precision_recall(cm_dict[k])
	return pr_dict,cm_dict


def pr_dict2pandas(pr_dict):
	'''Create pandas datafrom from the precision recall dict.'''
	names = ['r0','p0','f1_0','r1','p1','f1_1','mcc','fpr_0','fpr_1']
	df = pd.DataFrame.from_dict(pr_dict,orient='index')
	df.columns = names
	df['threshold'] = df.index
	return df


def plot_roc(pr_dict):
	'''Plot the receiver operator curce based on the precision recall dict (which also contains the fals positive rate).'''
	d = pr_dict2pandas(pr_dict)
	f = plt.figure()
	plt.plot(d.fpr_0,d.r0)
	plt.plot(d.fpr_1,d.r1)
	plt.plot(d.fpr_0,d.r0,'ro')
	plt.plot(d.fpr_1,d.r1,'bo')
	for i in range(d.shape[0]):
		plt.annotate(d.threshold.iloc[i], xy=(d.fpr_0.iloc[i],d.r0.iloc[i]))
		plt.annotate(d.threshold.iloc[i], xy=(d.fpr_1.iloc[i],d.r1.iloc[i]))
	plt.ylabel('true positive rate / recall')
	plt.xlabel('false positive rate')
	plt.legend(('clean','artifact'))
	plt.title('ROC curve eeg data with thresholding')
	plt.grid()
	fmmc = plt.figure()
	plt.plot(d.threshold,d.mcc)
	plt.plot(d.threshold,d.mcc)
	plt.title('Matthews correlation coefficient as a function of threshold')
	plt.grid()
	return f

def plot_pr_dict(pr_dict):
	'''Plots precision and recall against each other, might not be usefull.'''
	d = pr_dict2pandas(pr_dict)
	f = plt.figure()
	plt.plot(d.r0,d.p0)
	plt.plot(d.r1,d.p1)
	plt.plot(d.r0,d.p0,'ro')
	plt.plot(d.r1,d.p1,'bo')
	for i in range(d.shape[0]):
		plt.annotate(d.threshold.iloc[i], xy=(d.r0.iloc[i],d.p0.iloc[i]))
		plt.annotate(d.threshold.iloc[i], xy=(d.r1.iloc[i],d.p1.iloc[i]))
	plt.ylabel('precision')
	plt.xlabel('recall')
	plt.legend(('clean','artifact'))
	plt.title('Precision-Recall eeg data with thresholding')
	plt.grid()
	fmmc = plt.figure()
	plt.plot(d.threshold,d.mcc)
	plt.plot(d.threshold,d.mcc)
	plt.title('Matthews correlation coefficient as a function of threshold')
	plt.grid()
	return f

	

def matthews_correlation_coefficient(cm):
	'''Calculate the mcc based on the confusion matrix.
	-1 perfect disagreement of predicted and ground truth, 0 random agrement, 1 perfect agreement.
	'''
	print(cm.dtype)
	if cm.shape != (2,2): 
		print('confusion matrix should be 2X2 is:',cm.shape)
		return 0
	tp = int(cm[0,0])
	tn = int(cm[1,1])
	fp = int(cm[0,1])
	fn = int(cm[1,0])
	numerator = (tp*tn - fp*fn) 
	# if any of the sums in the denominator == 0, set arbitrarily to 1, this will result in a mcc of 0
	if tp+fp == 0 or tp+fn == 0 or tn+fp == 0 or tn+fn ==0: denominator = 1
	else:denominator = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) ** 0.5
	print(numerator, denominator,tp,tn,fp,fn)
	return numerator/denominator
	
	
def make_all_threshold_indices(thresholds = None):
	'''Set artifact-clean according to the thresholds for each epoch in data structure.
	thresholds is a list of thresholds
	If you need the output dict, consider loading it.'''
	output_dict = {}
	if thresholds == None: thresholds = range(40, 201,10)
	fn_indices = glob.glob(path.snippet_annotation+ '*gt_indices*')
	fn_data = glob.glob(path.artifact_training_dataraw + 'pp*data.npy')
	for fi in fn_indices:
		print(fi)
		fd = fni2fnd(fi,fn_data)
		if not fd: continue
		gt_indices = load_data(fi)
		print(len(np.where(gt_indices == 1)[0]),'artifacts')
		print(len(np.where(gt_indices == 0)[0]),'clean')
		print(len(np.where(gt_indices == 2)[0]),'other')
		d = load_data(fd) 
		d = hamming_data(d)
		threshold_indices = [make_threshold_indices(d,t) for t in thresholds]
		for i,ti in enumerate(threshold_indices):
			print(fi,thresholds[i])
			cm,report,mcc = compare_artifacts_groundtruth(gt_indices,ti)
			output_fn = make_indices_output_filename(fi,thresholds[i])
			save_indices(output_fn,ti)
			output_dict[fi,thresholds[i]] = [cm,report,mcc]
		print('-'*80+'\n')
	return output_dict

			
		
def load_output_dict():
	'''The output dict is made during the computation of min-max thresholding, it faster to load it.'''
	fin = open(path.snippet_annotation + 'comparison_dict','rb')
	return pickle.load(fin)





		
