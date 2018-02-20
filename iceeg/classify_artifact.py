import copy
import cnn_data
import cnn_output_data as cod
import model_cnn
import model_cnn_output as mco
import numpy as np
import path
import sklearn.metrics
import tensorflow as tf
import utils
import windower

class classify_artifact:
	'''Classify the artifacts and clean parts of a eeg block.'''
	def __init__(self,name = None,model_cnn_name = 'rep-3_perc-50_fold-2_part-90' , mco_name = 'cnn_output_model_wl-201_perc-93',fo = None,block = None,make_adj_pred = False):
		tf.reset_default_graph()
		self.fo = fo
		if not self.set_block_name(block,name): return None
		self.model_cnn_name = model_cnn_name
		self.mco_name = mco_name
		if make_adj_pred:
			self.load_predicted()
			self.generate_predicted_adj()



	def __str__(self):
		m = 'name:\t\t\t' + str(self.name) + '\n'
		m += 'model_cnn_name:\t\t' + str(self.model_cnn_name) + '\n'
		m += 'mco:\t\t\t' + str(self.mco_name) + '\n'
		if hasattr(self,'predicted') and isinstance(self.predicted,np.ndarray): m += 'predicted:\t\tavailable\n'
		else: m += 'predicted:\t\tNA\n'
		if hasattr(self,'predicted_adj') and isinstance(self.predicted_adj,np.ndarray): m += 'predicted_adj:\t\tavailable\n'
		else: m += 'predicted_adj:\t\tNA\n'
		if hasattr(self,'predicted_perc') and isinstance(self.predicted_perc,np.ndarray): m += 'predicted_perc:\t\tavailable\n'
		else: m += 'predicted_perc:\t\tNA\n'
		return m


	def __repr__(self):
		return 'classify-artifact: ' +self.name.replace('_',' ')



	def load_predicted(self):
		'''Load artifact classifications.'''
		self.do = cod.cnn_output2data(name= self.name, model_name = self.model_cnn_name)
		self.predicted = self.do.predicted


	def generate_predicted(self, clean_up = True, save = True):
		'''Classify eeg data on artifacts.'''
		d = cnn_data.cnn_data(2) # fold 2 was used for training, 10 fold training and testing was computationally to expensive
		self.m = model_cnn.load_model(path.model + self.model_cnn_name,d)
		self.predicted_class, self.predicted_perc = self.m.predict_block(self.block, save = save)
		# self.predicted = self.predicted_perc[:,1]
		self.load_predicted()
		self.m.clean_up()


	def generate_predicted_adj(self, identifier = '', save = True):
		'''Adjust predicted artifacts generated by model_cnn with mco model.
		compare with ground trouth if available.
		'''
		self.predicted_adj = copy.copy(self.do.pc)

		if self.do.predicted_artifact_data.shape[0] > 0:
			self.mo = mco.load_model(path.cnn_output_data + self.mco_name,self.do)
			self.predicted_artifact_adj = self.mo.compute_prediction_class(data=self.do.predicted_artifact_data)
			self.predicted_adj[self.do.predicted_artifact_indices] = self.predicted_artifact_adj
			self.mo.clean_up()

		if not isinstance(self.do.ground_truth,np.ndarray) or np.max(self.do.ground_truth) >1 : self.confusion_matrix_adj = np.zeros((2,2))
		else: self.confusion_matrix_adj = sklearn.metrics.confusion_matrix(self.do.ground_truth,self.predicted_adj, labels = [0,1])
		self.output_name = path.snippet_annotation+ identifier + self.model_cnn_name + '_' + self.name 
		self.eval_name = path.confusion_matrices + identifier + self.model_cnn_name + '_' + self.name 
		if save:
			np.save(self.output_name + '_class-adj', self.predicted_adj)
			if isinstance(self.do.ground_truth, np.ndarray):
				np.save(self.eval_name+ '_cm', self.do.confusion_matrix)
				np.save(self.eval_name+ '_cm-adj', self.confusion_matrix_adj)
			else:print('no ground truth for',self.name)


	def set_block_name(self,block,name):
		'''Sets name and loads corresponding block object.'''
		if name == None and block == None:
			print('Please provide name or block object.')
			return False
		if name == None:
			self.name = windower.make_name(block)
			self.block = block
		if block == None:
			self.block = utils.name2block(name,self.fo)
			self.name = name
		return True


def get_names_output_files(model_name = 'rep-3_perc-50_fold-2_part-90'):
	'''Get the filenames of the perc files, which are output from a cnn model.

	model_name  	name of the cnn model to generate the output files.
	'''
	fn = glob.glob(path.snippet_annotation + model_name + '_pp*class.npy')
	names = [f.split('part-90_')[-1].split('_class.npy')[0] for f in fn]
	return names



