import copy
import peakutils
import matplotlib.pyplot as plt
import os
import path
import pickle

plt.ion()

class Blinks:
	'''Contain blink information based on VEOG channel from raw mne object.
	'''

	def __init__(self,raw = None,  pre = 200, post = 300, thres = 0.45, min_dist = 200, plot = True, block= None):
		'''Extract blink information from VEOG channel from raw mne object.

		raw 		mne object, loaded eeg data (veog channel should be created and named VEOG
		pre 		n sample before peak detection (positive)
		post 		n sample after peak detection (positive)
		thres   	threshold for peak detection with peakutils (between 0 - 1 (perc from maximum value)
		min_dist 	minimum distance between peaks
		plot 		whether to plot results
		block 		load blink info from file, raw file takes precedence (if raw provided file not loaded)
		'''
		if raw == None and not block:
			print('please specify to load file or provide raw eeg data file')
			return 0 
		if not block:
			if 'VEOG' not in raw.ch_names:
				print('No VEOG channel found in the raw data object, please create eog channels')
				return 0
			self.raw_filename = raw.info['filename']
			self.st_sample = raw.first_samp
			self.n_samples = len(raw)
			self.pre = pre
			self.post = post
			self.thres = thres
			self.min_dist = min_dist
			self.find_peaks()
		else: self.load_blinks(block)
		if plot: self.plot()

	def __str__(self):
		m = 'n blinks:\t\t' + str(self.nblinks) + '\n'
		m += 'unusable samples:\t\t' + str(self.unusable_samples) + '\n'
		m += 'perc_unusable:\t\t' + str(self.perc_unusable) + '\n'
		m += 'eeg filename:\t\t' + self.raw_filename + '\n'
		m += 'has veog:\t\t' + str(hasattr(self,'veog'))
		return m

	def extract_veog(self,raw):
		'''Extract veog channel from raw mne eeg data object.'''
		if self.raw_filename != raw.info['filename']:
			print('filename of eeg data does not correspond with original:',raw.info['filename'],self.raw_filename)
			return 0
		self.ch_index = raw.ch_names.index('VEOG')
		# load veog and convert to micro volts ( raw object store values as volts)
		self.veog = raw[self.ch_index][0][0] 


	def find_peaks(self):
		'''Find peaks in veog channel and define start and end points of eog epoch.

		thres determines minimum value of peak
		min_dist determines minimum distance between peaks
		pre and post determine length of blink epoch relative to peak
		'''
		self.peaks = peakutils.indexes(self.veog,self.thres,self.min_dist)
		self.values = self.veog[self.peaks] * 10 ** 6
		pre, post = self.pre,self.post
		# create start point by subtracting pre samples from peak index
		self.start = [p - pre if p > pre else 0 for p in self.peaks]
		# create end point by adding post samples from peak index
		self.end = [p + post if p < len(self.veog) + post else len(sel.veog) -1 for p in self.peaks]
		self.nblinks = len(self.peaks)
		self.unusable_samples = (self.pre+self.post) * self.nblinks 
		self.perc_unusable = round((self.unusable_samples / self.n_samples) * 100,2)


	def save_blinks(self):
		'''Save object to file with name == eeg file and extension .blinks in path.blinks folder.'''
		self.fn = self.raw_filename.split('/')[-1] + '.blinks'
		output = copy.deepcopy(self)
		if hasattr(output,'veog'): del output.veog
		fout = open(path.blinks + self.fn ,'wb')
		pickle.dump(output,fout,-1)


	def load_blinks(self,block):
		'''Load object to file with name == eeg file and extension .blinks in path.blinks folder.'''
		fn = path.blinks + block.vmrk.vmrk_fn.split('/')[-1].replace('.vmrk','.blinks')
		if not os.path.isfile(fn):
			print('File does not excist, please provide raw eeg data object')
			return 0
		fin = open(fn,'rb')
		self = pickle.load(fin)

		

	def plot(self, raw = None,skip_veog = False):
		'''plot blink epochs and veog channel.'''
		if raw: self.extract_veog(raw)
		t = 'Blinks:'+str(self.nblinks) + '\n\n'
		t +='bad samples:' + str(self.unusable_samples) + '      '
		t +=' perc: ' + str(self.perc_unusable) + '      '
		self.fig = plt.figure()
		self.fig.canvas.set_window_title(self.raw_filename.split('/')[-1])
		plt.title(t)
		if hasattr(self,'veog') and not skip_veog: plt.plot(self.veog,color = 'grey')
		plt.plot(self.peaks,self.values,'ro')
		[plt.axvline(s,color='tomato',linestyle='-',linewidth=1) for s in self.start]
		[plt.axvline(e,color='tomato',linestyle='--',linewidth=1) for e in self.end]
