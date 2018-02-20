import glob
import os
import pandas as pd

class alog:
	'''Aggregate information about start/end times in the experiment.

	.log Panda df containing:
		-filenames of the experimental audio files
		-participant number and age (should add gender)
		(should be extended with answer accuracy and temperature and humidety)
	'''
	
	def __init__(self, directory = '', filename = '', save_new_dir = True):
		'''Aggregate information about start/end times of events in the experiment.
		
		Keywords:
		pp_id = participant id (1-48) int
		exp_type = experimental type (k/o/ifadv) reflects register of speech in audio file str
		'''
		if directory[-1] != '/': directory += '/'
		self.directory = directory
		self.filename = filename
		self.save_new_dir = save_new_dir
		self.fn = []
		if self.directory:
			if os.path.isdir(self.directory):
				self.fn = glob.glob(self.directory + 'pp*.txt')
			else: print(self.directory,'does not exist')
		
		if self.fn: self.anonymize_directory()
		elif self.filename: self.anonymize(f)
		else: print('Please specify a name or directory to anonimize data')
		self.read_log()
		

	def __str__(self):
		m = 'log filename:\t\t' + str(self.filename) + '\n'
		m += 'directory:\t' + str(self.directory) + '\n'
		return m

	def anonymize_directory(self):
		for f in self.fn:
			print(f)
			self.anonymize(f)
		
	def anonymize(self,f):
		self.log_fn = f
		self.read_log()
		self.alog = [[line[0],'***'] + line[2:] for line in self.log if len(line) > 3]
		self.save_log()

	def read_log(self):
		self.log = [line.split(' ') for line in open(self.log_fn).read().split('\n') if line]

	def save_log(self):
		if not os.path.isdir(self.directory+ 'ANONYMIZE_LOG/') and self.save_new_dir: 
			print('making anonymizing dir')
			os.mkdir(self.directory + 'ANONYMIZE_LOG/')
		if self.save_new_dir: fout = open(self.directory +'/ANONYMIZE_LOG/' + self.log_fn.split('/')[-1],'w')
		else: fout = open(self.log_fn,'w')
	
		fout.write('\n'.join([' '.join(line) for line in self.alog]))
		fout.close()
		

