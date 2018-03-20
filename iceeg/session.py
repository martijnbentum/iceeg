'''Load information for 1 experimental session of 1 participant

Session class uses block log vmrk objects to load and store all information 
'''

import block
import copy
import glob
import log
import mne
import mne.preprocessing.ica as ica
import numpy as np
import pandas as pd
import path
import re
import utils
import vmrk


class Session:
	'''A session is 1 of three experiments the participant participated in

	Each session the participant heard speech from a different register
	(spontaneous dialogies (ifadv) news broadcast (k) read aloud stories (o)
	
	Each session heard multiple audio files with speech from the register
	Info about that links words in the audio files to EEG markers can be found
	in .blocks list, this is a list of block object, and can also be accessed
	as .b1 ... .bn.

	ifadv 6 blocks / o 7 blocks / k 21 blocks (k blocks are much shorter)

	After audio file presentation the pp answered (yes/no) comprehension questions 
	(accuracy should be added

	The .log object contains logging info from Presentation and answer data
		also contains several dictionaries to map markers / file ids to other information
	The .vmrk object contains BrainVision marker data
	'''

	def __init__(self, pp_id = None,exp_type = None,fid2ort = None):
		'''Load information for 1 experimental session of 1 participant

		Keywords:
		pp_id = participant number (1-48) int
		exp_type = experimental type (o/k/ifadv) reflect register of audio file
		fid2ort = dictionary that maps file id to ort object (use deepcopy for each pp)
		path = default is set to the parent directory
		'''

		print('loading session with:',pp_id,exp_type)
		self.artifact_perc = 0.0
		self.pp_id = pp_id
		self.exp_type = exp_type
		self.experiment_name = utils.exptype2explanation_dict[self.exp_type]
		self.fid2ort = fid2ort
		self.log = log.log(self.pp_id,self.exp_type)
		self.session = self.log.session
		self.vmrk = vmrk.vmrk(self.pp_id,self.exp_type)
		self.n_eeg_recordings = self.vmrk.n_eeg_recordings
		self.set_start_end_times() # start end times experiment
		self.nblocks = self.log.log['block'].values[-1]
		self.load_blocks()
		utils.make_attributes_available(self,'b',self.blocks)
		self.eeg_loaded = False


	def __str__(self):
		m = '\nSESSION OBJECT\n'
		m += 'Participant number:\t' + str(self.pp_id) + '\n'
		m += 'Experiment name:\t' + str(self.exp_type) + ' ' + self.experiment_name + '\n'
		m += 'Session number: \t' + str(self.session) + '\n'
		m += 'nblocks:\t\t' + str(self.nblocks) + '\n'
		m += 'Start experiment:\t' + str(self.start_exp) + '\n'
		m += 'End experiment:\t\t' + str(self.end_exp) + '\n'
		m += 'Duration:\t\t' + str(self.duration).split(' ')[-1] + '\n'
		m += 'n words:\t\t' + str(self.nwords) + '\n'
		m += 'n content words:\t' + str(self.ncontent_words) + '\n'
		m += 'n blocks:\t\t' + str(self.nblocks) + '\n'
		m += 'n eeg_recordings:\t' + str(self.n_eeg_recordings) + '\n'
		m += 'nartifacts\t\t'+str(self.nartifacts) + '\n'
		m += 'total dur\t\t'+str(int(self.total_duration)) + '\n'
		m += 'total artifact dur\t'+str(int(self.total_artifact_duration)) + '\n'
		m += 'artifact_perc\t\t'+str(round(self.artifact_perc,3)) + '\n'
		m += self.vmrk.__str__()
		m += self.log.__str__()
		return m

	def __repr__(self):
		return 'session-object:\t' + self.exp_type + ': ' + self.experiment_name + '\tpp ' + str(self.pp_id) + '\tnwords: ' + str(self.nwords) + '\tartifact perc: ' + str(round(self.artifact_perc,3))


	def set_start_end_times(self):
		'''Set the start and end date of the experiment and calculates duration.'''
		if not self.log.log_fn:
			return None
		self.start_exp = self.log.start_exp 
		self.end_exp = self.log.end_exp
		self.duration = self.log.duration 

		
	def load_blocks(self):
		'''Create a block object for each audio file in the experiment.'''
		self.blocks = []
		self.nwords = 0
		self.ncontent_words = 0
		self.nblinks = 0
		self.nartifacts,self.total_duration,self.total_artifact_duration = 0,0,0
		for i in range(1,self.nblocks+1):
			self.blocks.append(block.block(self.pp_id,self.exp_type,self.vmrk,self.log,i,self.fid2ort))
			self.nwords += self.blocks[-1].nwords
			self.ncontent_words += self.blocks[-1].ncontent_words
			if self.blocks[-1].nblinks != 'NA': self.nblinks += self.blocks[-1].nblinks
			try:
				self.nartifacts += self.blocks[-1].nartifacts
				self.total_duration += self.blocks[-1].duration_sample/1000
				self.total_artifact_duration += self.blocks[-1].total_artifact_duration
			except: print('block object has not artifacts.')
			if self.total_duration == 0: self.artifact_perc = 0
			else: self.artifact_perc = self.total_artifact_duration / self.total_duration
		self.nblocks = len(self.blocks)

	def make_blocks_available(self):
		'''make block available on object as a property .b1 .b2 .b3 etc.'''
		self.block_property_names = ['b' +str(i) for i in range(1,self.nblocks + 1)]
		[setattr(self,bname,self.blocks[i]) for i,bname in enumerate(self.block_property_names)]


	def print_block_info(self):
		'''Print information of each block.'''
		for b in self.blocks:
			print(b)


	def load_eeg_data(self,freq = [0.05,30]):
		'''Loads eeg data of each block and concatenates it into self.raw.'''
		print('Loading all data from session and concatenating into one raw object.')
		good_indices = []
		for i,b in enumerate(self.blocks):
			print(b.bid)
			b.load_eeg_data(freq=freq)
			if b.artifacts != 'NA' and b.artifacts != []:
				b.raw.annotations = mne.Annotations(b.start_artifacts,b.duration_artifacts,'BAD')
			if b.eeg_loaded: good_indices.append(i)
		if len(good_indices) == 0: return 0
		self.n_blocks_loaded = str(len(good_indices))
		self.raw = self.blocks[good_indices[0]].raw
		for i in good_indices[1:]:
			b = self.blocks[i]
			self.raw.append(b.raw)
			b.unload_eeg_data()
		self.eeg_loaded = True

	def unload_eeg_data(self):
		'''Unload eeg data, deletes self.raw.'''
		if hasattr(self,'raw'):
			delattr(self,'raw')
		self.eeg_loaded = False


	def fit_ica(self):
		'''Fit ica on session object. All block data is loaded and concatenated in self.raw.
		EEG data is bandpassed filtered on 1-30 Hz, the ica solution can be used on data
		without or different bandpass filter see:
		Winkler et al.
		On the influence of high-pass filtering on ICA-based artifact reduction in EEG-ERP
		WIP: extend to be able to set ica approach (with or without artifact rejection, filename)
		'''
		print('Fitting ica on all session data.')
		if not self.eeg_loaded: 
			try:self.load_eeg_data(freq = [1,30])
			except: return 0
			if not self.eeg_loaded: return 0
		self.ica = ica.ICA()
		self.ica_e = ica.ICA()
		self.ica.fit(self.raw,reject_by_annotation = False)
		self.eog_comp, self.eog_scores = self.ica.find_bads_eog(self.raw,reject_by_annotation = False)
		self.ica.save(path.ica_solutions + 'pp' + str(self.pp_id) +'_exp-'+self.exp_type + '_session_all-data-ica.fif')

		if len(self.raw.annotations) > 0:
			self.ica_e.fit(self.raw,reject_by_annotation = True)
			self.eog_comp_e, self.eog_scores_e = self.ica_e.find_bads_eog(self.raw,reject_by_annotation = False)
			self.ica_e.save(path.ica_solutions + 'pp' + str(self.pp_id) +'_exp-'+self.exp_type + '_session_no-artifact-ica.fif')
		else: self.eog_comp_e,self.eog_scores_e = self.eog_comp, self.eog_scores
		
		self.write_eog()


	def write_eog(self):
		'''Write text file with correlation between eog channels and ic from ica solution.
		performs ica with and without artifact rejection
		WIP: assumes that both all_data and artifact rejection is performed, 
		should be more general.
		'''
		if len(self.raw.annotations) == 0: xml_ok = 'xml_absent'
		else: xml_ok = 'xml_present'
		eog = ','.join(map(str,self.eog_comp))
		veog_scores = ','.join([str(self.eog_scores[0][i]) for i in self.eog_comp])
		heog_scores = ','.join([str(self.eog_scores[1][i]) for i in self.eog_comp])

		eog_e = ','.join(map(str,self.eog_comp_e))
		veog_scores_e = ','.join([str(self.eog_scores_e[0][i]) for i in self.eog_comp_e])
		heog_scores_e = ','.join([str(self.eog_scores_e[1][i]) for i in self.eog_comp_e])

		output = '\t'.join([eog,veog_scores,heog_scores,eog_e,veog_scores_e,heog_scores_e,xml_ok,str(self.artifact_perc),self.n_blocks_loaded,str(self.nblinks)])
		fout = open(path.ica_solutions + 'pp' + str(self.pp_id) + '_exp-'+self.exp_type + '_session_eog.txt','w')
		fout.write(output)
		fout.close()
		

