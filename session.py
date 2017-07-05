'''Load information for 1 experimental session of 1 participant

Session class uses block log vmrk objects to load and store all information 
'''

import block
import copy
import glob
import log
import numpy as np
import pandas as pd
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

	def __init__(self, pp_id = None,exp_type = None,fid2ort = None,path = '../'):
		'''Load information for 1 experimental session of 1 participant

		Keywords:
		pp_id = participant number (1-48) int
		exp_type = experimental type (o/k/ifadv) reflect register of audio file
		fid2ort = dictionary that maps file id to ort object (use deepcopy for each pp)
		path = default is set to the parent directory
		'''

		print('loading session with:',pp_id,exp_type,path)
		self.path = path
		self.pp_id = pp_id
		self.exp_type = exp_type
		self.fid2ort = fid2ort
		self.log = log.log(self.pp_id,self.exp_type,self.path)
		self.session = self.log.session
		self.vmrk = vmrk.vmrk(self.pp_id,self.exp_type,self.path)
		self.n_eeg_recordings = self.vmrk.n_eeg_recordings
		self.set_start_end_times() # start end times experiment
		self.nblocks = self.log.log['block'].values[-1]
		self.load_blocks()
		utils.make_attributes_available(self,'b',self.blocks)


	def __str__(self):
		m = '\nEXPERIMENT OBJECT\n'
		m += 'Participant number:\t' + str(self.pp_id) + '\n'
		m += 'Experiment name:\t' + str(self.exp_type) + '\n'
		m += 'Session number: \t' + str(self.session) + '\n'
		m += 'nblocks:\t\t' + str(self.nblocks) + '\n'
		m += 'Start experiment:\t' + str(self.start_exp) + '\n'
		m += 'End experiment:\t\t' + str(self.end_exp) + '\n'
		m += 'Duration:\t\t' + str(self.duration).split(' ')[-1] + '\n'
		m += 'n words:\t\t' + str(self.nwords) + '\n'
		m += 'n blocks:\t\t' + str(self.nblocks) + '\n'
		m += 'n eeg_recordings:\t' + str(self.n_eeg_recordings) + '\n'
		m += self.vmrk.__str__()
		m += self.log.__str__()
		return m


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
		for i in range(1,self.nblocks+1):
			self.blocks.append(block.block(self.pp_id,self.exp_type,self.vmrk,self.log,i,self.fid2ort))
			self.nwords += self.blocks[-1].nwords
		self.nblocks = len(self.blocks)

	def make_blocks_available(self):
		'''make block available on object as a property .b1 .b2 .b3 etc.'''
		self.block_property_names = ['b' +str(i) for i in range(1,self.nblocks + 1)]
		[setattr(self,bname,self.blocks[i]) for i,bname in enumerate(self.block_property_names)]


	def print_block_info(self):
		for b in self.blocks:
			print(b)

