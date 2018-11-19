'''Load information for 1 experimental session of 1 participant

Session class uses block log vmrk objects to load and store all information 
'''

import block
import copy
import eog
import glob
import handle_ica
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
		self.name = 'pp' + str(pp_id) + '_exp-' + exp_type
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
		m += 'n content words:\t' + str(self.ncwords) + '\n'
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
		return 'session-object: ' + self.exp_type + ' ' + self.experiment_name + '\tpp ' + str(self.pp_id) + '\tnwords: ' + str(self.nwords) + '\tncwords: ' + str(self.ncwords) + '\tartifact perc: ' + str(round(self.artifact_perc,3))


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
		self.nallwords = 0
		self.nwords = 0
		self.ncontent_words = 0
		self.ncwords = 0
		self.nblinks = 0
		self.remove_ch = []
		self.usability = []
		self.nartifacts,self.total_duration,self.total_artifact_duration = 0,0,0
		for i in range(1,self.nblocks+1):
			self.blocks.append(block.block(self.pp_id,self.exp_type,self.vmrk,self.log,i,self.fid2ort))
			self.ncontent_words += self.blocks[-1].ncontent_words
			if self.blocks[-1].nblinks != 'NA': self.nblinks += self.blocks[-1].nblinks
			try:
				self.nallwords += self.blocks[-1].nallwords
				if self.blocks[-1].xml.usability not in ['bad','doubtfull']:
					self.nwords += self.blocks[-1].nwords
					self.nartifacts += self.blocks[-1].nartifacts
					self.total_duration += self.blocks[-1].duration_sample/1000
					self.total_artifact_duration += self.blocks[-1].total_artifact_duration
					self.ncwords += self.blocks[-1].ncwords
				self.remove_ch.append(self.blocks[-1].xml.remove_ch)
				self.usability.append(self.blocks[-1].xml.usability)
			except: 
				print('block object has no artifacts.')
				self.remove_ch.append('NA')
				self.usability.append('NA')
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


	def load_eeg_data(self,freq = [0.05,30], block_indices = [], use_all=False ):
		'''Loads eeg data of each block and concatenates it into self.raw.
		freq 		low and high frequency of bad pass butterworth iir filter
		'''
		if use_all:
			print('Loading all data from session and concatenating into one raw object.')
		else:
			print('Loading data from blocks with the following indices:',block_indices)
		good_indices = []
		self.rejected_artifact_blocks = []
		self.not_loaded_eeg= []
		self.loaded_eeg= []
		for i,b in enumerate(self.blocks):
			if i not in block_indices and not use_all: 
				print('skipping block, with index:',i,'only loading:',block_indices)
				continue
			print(b.bid)
			b.load_eeg_data(freq=freq)
			if b.artifacts != 'NA' and b.artifacts != []:
				b.raw.annotations = mne.Annotations(b.start_artifacts,b.duration_artifacts,'BAD')
			if b.eeg_loaded: 
				good_indices.append(i)
				self.loaded_eeg.append(b)
			else: self.not_loaded_eeg.append(b)
		if len(good_indices) == 0: return 0
		self.n_blocks_loaded = str(len(good_indices))
		if self.n_blocks_loaded == 0:
			print('no data loaded')
			return
		self.raw = self.blocks[block_indices[0]].raw
		for i in block_indices[1:]:
			b = self.blocks[i]
			self.raw.append(b.raw)
			b.unload_eeg_data()
		self.eeg_loaded = True
		if use_all: self.current_block_indices = good_indices
		else: self.current_block_indices = block_indices
		if hasattr(self,'eog'): self.raw.info['bads'] = self.eog.rejected_channels

	def unload_eeg_data(self):
		'''Unload eeg data, deletes self.raw.'''
		if hasattr(self,'raw'):
			delattr(self,'raw')
		self.eeg_loaded = False


	def group_blocks(self):
		self.bad_block_indices=[i for i, x in enumerate(self.usability) if x in ['bad','doubtfull','NA']]
		temp = [tuple(line) for line in self.remove_ch]
		unique = list(set(temp))
		self.bindices = []
		for s in unique:
			ti = []
			for i, line in enumerate(temp):
				if line == s and i not in self.bad_block_indices:
					ti.append(i)
			if ti != []:
				self.bindices.append(ti)
		
	def fit_ica_grouped_blocks(self,reject_artifacts = True):
		self.group_blocks()
		for block_indices in self.bindices:
			reject_channels = self.remove_ch[block_indices[0]]
			print('loading block indices:',block_indices)
			print('bad channels:',reject_channels)
			self.unload_eeg_data()
			self.fit_ica(block_indices= block_indices,reject_channels = reject_channels)
			

	def check_ica_grouped_blocks(self, rejected_artifacts = True):
		self.group_blocks()
		for block_indices in self.bindices:
			print('loading blocks with indices:',block_indices)
			self.current_block_indices = block_indices
			self.load_ica(rejected_artifacts = rejected_artifacts)
			print('loading eog with filename:',self.eog_filename)
			print('plotting ica topoplot')
			self.plot_ica()
			a = input('please correct eog')
			print('loading next eog')
		print('checked_all_eogs')


	def fit_ica(self, reject_artifacts= True,reject_channels = ['Fp2'],block_indices = []):
		handle_ica.fit(self, reject_artifacts= reject_artifacts,reject_channels = reject_channels,block_indices = block_indices)


	def load_ica(self, rejected_artifacts = True,filename_ica = ''):
		handle_ica.load(self, rejected_artifacts = rejected_artifacts,filename_ica = filename_ica)


	def plot_ica(self):
		handle_ica.plot(self)


	def create_eog(self):
		'''Create eog object with correlation between ICA components and eog channels.
		save it to xml file
		'''
		if not hasattr(self,'raw'): self.load_eeg_data()
		self.eog_comp, self.eog_scores = self.ica.find_bads_eog(self.raw,reject_by_annotation = False)
		lbid = [b.bid for b in self.loaded_eeg]
		rbid = [b.bid for b in self.rejected_artifact_blocks]
		nlbid = [b.bid for b in self.not_loaded_eeg]
		self.eog = eog.eog(scores = self.eog_scores, comps= self.eog_comp,b = self, ica_filename = self.ica_filename, filename = self.ica_filename.replace('ica.fif','eog.xml'),name = self.make_name(),rejected_channels = self.ica_rejected_channels,loaded_bid = lbid, rejected_bid = rbid, not_loaded_bid = nlbid)
		self.eog.write()

		
	def make_name(self):
		b_names = '-'.join(['b' + str(i+1) for i in self.current_block_indices])
		return 'pp' + str(self.pp_id) + '_' + self.exp_type + '_' + b_names


def reload(s):
	'''Reload a session object for testing and debugging. '''
	return Session(s.pp_id,s.exp_type,s.fid2ort)

