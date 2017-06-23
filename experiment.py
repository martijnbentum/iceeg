import block
import glob
import log
import numpy as np
import pandas as pd
import re
import vmrk
PATH = '../'


class experiment:
	def get_id(self):
		return self.pp_id

	def __init__(self, pp_id = None,exp_type = None,path = None):
		print('loading experiment with:',pp_id,exp_type,path)
		if not path: self.path = PATH
		else: self.path = path
		self.pp_id = pp_id
		self.exp_type = exp_type
		self.log = log.log(self.pp_id,self.exp_type,self.path)
		self.session = self.log.session
		self.vmrk = vmrk.vmrk(self.pp_id,self.exp_type,self.path)
		self.n_eeg_recordings = self.vmrk.n_eeg_recordings
		self.set_start_end_times() # start end times experiment
		self.nblocks = self.log.log['block'].values[-1]
		self.load_blocks()


	def __str__(self):
		# print relevant info 
		m = '\nEXPERIMENT OBJECT\n'
		m += 'Participant number:\t' + str(self.pp_id) + '\n'
		m += 'Experiment name:\t' + str(self.exp_type) + '\n'
		m += 'Session number: \t' + str(self.session) + '\n'
		m += 'nblocks:\t\t' + str(self.nblocks) + '\n'
		m += 'Start experiment:\t' + str(self.start_exp) + '\n'
		m += 'End experiment:\t\t' + str(self.end_exp) + '\n'
		m += 'Duration:\t\t' + str(self.duration).split(' ')[-1] + '\n'
		m += 'n words:\t\t' + str(self.nwords) + '\n'
		# fields = ('pp_id exp_type session start_exp end_exp duration ' \
			# 'log vmrk blocks nblocks').split(' ')
		# m += 'FIELDS:\n' + '\t'.join(fields)
		m += '\n' + '-'*50 + self.vmrk.__str__()
		m += '\n' + '-'*50 + self.log.__str__()
		return m


	def set_start_end_times(self):
		# sets the start and end date of the experiment and calculates duration
		if not self.log.log_fn:
			return None
		self.start_exp = self.log.start_exp 
		self.end_exp = self.log.end_exp
		self.duration = self.log.duration 

		
	def load_blocks(self):
		self.blocks = []
		self.nwords = 0
		for i in range(1,self.nblocks+1):
			self.blocks.append(block.block(self.pp_id,self.exp_type,self.vmrk,self.log,i))
			self.nwords += self.blocks[-1].nwords
