import copy
import peakutils
import matplotlib.pyplot as plt
import numpy as np
import os
import path
import pickle

plt.ion()

class Blinks:
	'''Contain blink information based on VEOG channel from raw mne object.
	'''

	def __init__(self,raw = None,  pre = 200, post = 300, thres_value = 60, min_dist = 200, plot = False, block= None,marker='unk',remove_veog = True):
		'''Extract blink information from VEOG channel from raw mne object.

		raw 		mne object, loaded eeg data (veog channel should be created and named VEOG
		pre 		n sample before peak detection (positive)
		post 		n sample after peak detection (positive)
		thres_value threshold for peak detection in mu volts 
		min_dist 	minimum distance between peaks
		plot 		whether to plot results
		block 		load blink info from file, raw file takes precedence (if raw provided file not loaded)
		'''
		if raw == None and not block:
			print('please specify to load file or provide raw eeg data file')
			return 0 
		if not block:
			print('detecting blinks with peak detection.')
			if 'VEOG' not in raw.ch_names:
				print('No VEOG channel found in the raw data object, please create eog channels')
				return 0
			self.raw_filename = raw.info['filename']
			self.st_sample = raw.first_samp
			self.n_samples = len(raw)
			self.pre = pre
			self.post = post
			self.thres_value = thres_value
			self.min_dist = min_dist
			self.marker = marker
			self.remove_veog = remove_veog
			self.extract_veog(raw)
			self.find_peaks()
			self.save_blinks()
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
		self.veog = raw[self.ch_index][0][0] * 10 ** 6


	def find_peaks(self):
		'''Find peaks in veog channel and define start and end points of eog epoch.

		thres determines minimum value of peak
		min_dist determines minimum distance between peaks
		pre and post determine length of blink epoch relative to peak
		'''
		self.thres = (self.thres_value + abs(self.veog.min())) / (self.veog.max() - self.veog.min())
		print(self.thres,'threshold',self.thres_value,'value')
		self.peaks = peakutils.indexes(self.veog,self.thres,self.min_dist)
		self.values = self.veog[self.peaks] 
		pre, post = self.pre,self.post
		# create start point by subtracting pre samples from peak index
		self.start = [p - pre if p > pre else 0 for p in self.peaks]
		# create end point by adding post samples from peak index
		self.end = [p + post if p < len(self.veog) + post else len(self.veog) -1 for p in self.peaks]
		self.nblinks = len(self.peaks)
		self.unusable_samples = (self.pre+self.post) * self.nblinks 
		self.perc_unusable = round((self.unusable_samples / self.n_samples) * 100,2)
		self.validate_peaks()
		# self.prune_peaks()
		self.find_start_end()

	def find_start_end(self):
		self.start_peaks = np.array([],dtype=int)
		self.end_peaks = np.array([],dtype=int)
		self.start_slope = np.zeros(len(self.peaks))
		self.end_slope = np.zeros(len(self.peaks))
		for i,p in enumerate(self.peaks):
			if p > 500: start_slope = self.veog[p-500:p] * - 1
			else: start_slope = self.veog[0:p] * - 1
				
			if p < len(self.veog) - 801: end_slope = self.veog[p:p+800] * - 1
			else: end_slope = self.veog[p:-1] * - 1
				

			if len(start_slope) > 100:
				sp = peakutils.indexes(start_slope,0.0001,1) + p-500
				self.start_peaks = np.append(self.start_peaks, sp )
				if len(sp) > 1:
					sp = np.append(sp,np.array(p))
					self.start_slope[i] = max(np.diff(self.veog[sp]) ) 
			else: self.start_slope[i] = 0
			if len(end_slope) > 100:
				ep = peakutils.indexes(end_slope,0.0001,1) + p
				self.end_peaks = np.append(self.end_peaks, ep )
				if len(ep) > 1:
					ep = np.append(ep,np.array(p))
					self.end_slope[i] = max(np.diff(self.veog[ep]) ) 
			else: self.end_slope[i] = 0

			# print(self.veog[sp])
			# print(np.diff(self.veog[sp]))
			# print(max(np.diff(self.veog[sp])))
			# print('---')

		self.start_values = self.veog[self.start_peaks]
		self.end_values = self.veog[self.end_peaks]

	def zscore_peaks(self):
		self.zscores = np.zeros(len(self.peaks))
		for i,p in enumerate(self.peaks):
			if p > 600: start = p - 600
			else: start = 0	
			before = np.arange(start,p-200)
			if p < len(self.veog) - 600: end = p + 600
			else: end = len(self.veog) - 1
			after = np.arange(p + 300,end)

			indices = np.append(before,after)
			mean = self.veog[indices].mean() 
			std = self.veog[indices].std() 
			self.zscores[i] =  (self.values[i] - mean ) / std

	def validate_peaks(self):
		self.zscore_peaks()
		self.before_dif = np.zeros(len(self.peaks))
		self.after_dif = np.zeros(len(self.peaks))
		self.both_dif = np.zeros(len(self.peaks))
		self.auc = np.zeros(len(self.peaks))
		self.npeaks = np.zeros(len(self.peaks)) 
		for i,p in enumerate(self.peaks):

			if p >= self.pre:
				self.before_dif[i] = self.veog[p]  - self.veog[p-self.pre]  
				start = p - self.pre
			else:
				self.before_dif[i] = self.veog[p] - self.veog[0] 
				start = 0
			if p+ self.post  < len(self.veog):
				self.after_dif[i] = self.veog[p] - self.veog[p+self.post]  
				end = p + self.post
			else:
				self.after_dif[i] = self.veog[p] - self.veog[-1] 
				end = len(self.veog) -1

			blink = self.veog[start:end] - (self.before_dif[i] + self.after_dif[i]) /2
			# self.auc[i] = abs(blink).mean()
			# self.npeaks[i] = len(peakutils.indexes(self.veog[start:end],0.8,1))
			self.both_dif[i] = self.before_dif[i] + self.after_dif[i]


	def prune_peaks(self,z_threshold = 100):
		remove = np.where(self.both_dif< z_threshold)[0]
		self.peaks = np.delete(self.peaks,remove)
		self.values= np.delete(self.values,remove)
		self.zscores= np.delete(self.zscores,remove)
		# self.auc= np.delete(self.auc,remove)
		# self.npeaks= np.delete(self.npeaks,remove)
		self.before_dif= np.delete(self.after_dif,remove)
		self.after_dif= np.delete(self.after_dif,remove)
		self.both_dif= np.delete(self.both_dif,remove)
		self.start= np.delete(self.start,remove)
		self.end= np.delete(self.end,remove)
		self.nblinks = len(self.peaks)
		self.unusable_samples = (self.pre+self.post) * self.nblinks 
		self.perc_unusable = round((self.unusable_samples / self.n_samples) * 100,2)
	


	def save_blinks(self):
		'''Save object to file with name == eeg file and extension .blinks in path.blinks folder.'''
		self.fn = self.raw_filename.split('/')[-1].strip('.eeg')+ '_' + str(self.marker) + '.blinks'
		output = copy.deepcopy(self)
		if hasattr(output,'veog') and self.remove_veog: del output.veog
		fout = open(path.blinks + self.fn ,'wb')
		pickle.dump(output,fout,-1)


	def load_blinks(self,block):
		'''Load object to file with name == eeg file and extension .blinks in path.blinks folder.'''
		fn = path.blinks + block.vmrk.vmrk_fn.split('/')[-1].strip('.vmrk') + '_' + str(block.marker) + '.blinks'
		if not os.path.isfile(fn):
			print('File does not excist, please provide raw eeg data object')
			return 0
		fin = open(fn,'rb')
		self = pickle.load(fin)

	def classify_blinks(self):
		self.responses = np.zeros(len(self.peaks),dtype=int)
		i = 0
		print(len(self.peaks), 'possible blinks found')
		self.cname = input('name: ')
		while 1:	
			p = self.peaks[i]
			
			t = 'Blinks:'+str(len(self.npeaks)) + '\n\n'
			t += 'current # ' + str(i+1)
			fig = plt.figure()
			fig.canvas.set_window_title(self.raw_filename.split('/')[-1])
			plt.title(t)
			start = p-500 if p > 500 else 0
			end = p+500 if p < (len(self.veog) - 800) else len(self.veog)
			plt.plot(self.veog[start:end])
			peak = p - start
			plt.plot(peak,self.veog[p],'ro')
			r = self.get_input(' (1:blinks   2:nothing   3:back) ')
			if r == '3': i -= 1
			else: 
				self.responses[i] = r
				i += 1
			plt.close(fig)
			if i == len(self.peaks) -1:
				break
		self.write_classification_output()
			
		
	def get_input(self,m):
		while 1:
			r = input(m)
			if r == '3': 
				return r
				print('going 1 blink backwards')
			elif r not in ['1','2']:
				print('please only use 1 or 2')
			else:
				return int(r)

	def write_classification_output(self):
		self.output = []
		for i,p in enumerate(self.peaks):
			l = map(str,[self.st_sample, self.raw_filename, self.marker,p,self.st_sample + p,self.responses[i]])
			self.output.append('\t'.join(l))
		fout = open(path.blinks + self.fn.strip('.blinks') + '_' + str(self.marker) + '_' + self.cname + '.classification','w')
		fout.write('\n'.join(self.output))
		fout.close()

	def plot(self, raw = None,skip_veog = False):
		'''plot blink epochs and veog channel.'''
		if raw: self.extract_veog(raw)
		t = 'Blinks:'+str(self.nblinks) + '\n\n'
		t +='bad samples:' + str(self.unusable_samples) + '      '
		t +=' perc: ' + str(self.perc_unusable) + '      '
		self.fig = plt.figure()
		self.fig.canvas.set_window_title(self.raw_filename.split('/')[-1])
		plt.title(t)
		if hasattr(self,'veog') and not skip_veog: plt.plot(self.veog  ,color = 'grey')
		plt.plot(self.peaks,self.values,'ro')
		plt.plot(self.peaks,self.zscores,'g^')
		# plt.plot(self.peaks,self.before_dif,'m.')
		plt.plot(self.peaks,self.both_dif,'b<')
		plt.plot(self.start_peaks,self.start_values,'b8')
		plt.plot(self.end_peaks,self.end_values,'cp')
		plt.plot(self.peaks,self.start_slope,'m>')
		# plt.plot(self.peaks,self.end_slope,'cv')
		for x,y in zip(self.peaks,self.zscores):
			plt.annotate(str(round(y,2)),xy=(x,y+0.5))
		for x,y in zip(self.peaks,self.start_slope):
			plt.annotate(str(round(y,2)),xy=(x,y+0.5))
		for x,y in zip(self.peaks,self.both_dif):
			plt.annotate(str(round(y,2)),xy=(x,y+0.5))
		[plt.axvline(s,color='tomato',linestyle='-',linewidth=1) for s in self.start]
		[plt.axvline(e,color='tomato',linestyle='--',linewidth=1) for e in self.end]
