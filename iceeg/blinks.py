import blink_model
import copy
import path
import peakutils
import matplotlib.pyplot as plt
import numpy as np
import os
import path
import pickle
import utils
import windower

plt.ion()

class Blinks:
	'''Contain blink information based on VEOG channel from raw mne object.
	'''

	def __init__(self,b,  pre = 200, post = 300, thres_value = 60, min_dist = 200, plot = False, marker='unk',remove_veog = True,force_create = False):
		'''Extract blink information from VEOG channel from raw mne object.

		b 			block object
		pre 		n sample before peak detection (positive)
		post 		n sample after peak detection (positive)
		thres_value threshold for peak detection in mu volts 
		min_dist 	minimum distance between peaks
		plot 		whether to plot results
		block 		load blink info from file, raw file takes precedence (if raw provided file not loaded)
		force_cre.. force create new blink object, do not load from file
		'''
		if force_create:blinks = 0
		else: blinks = load_blinks(b)
		if blinks == 0:
			self.name = windower.make_name(b)
			result = self.extract_veog(b)
			if result == 0: 
				self.nblinks = 'NA'
				return None
			print('detecting blinks with peak detection.')
			self.st_sample = b.st_sample
			self.n_samples = b.duration_sample
			self.pre = pre
			self.post = post
			self.thres_value = thres_value
			self.min_dist = min_dist
			self.marker = b.marker
			self.remove_veog = remove_veog
			if self.veog_loaded:
				self.find_peaks()
				self.save_blinks()
		else: self.__dict__.update(blinks.__dict__)
		if plot: self.plot()
		# if not hasattr(self,'blinks'):
		# self.peaks2model_data()
		# self.model_classification()
		# self.save_blinks()


	def __str__(self):
		m = 'n blinks:\t\t' + str(self.nblinks) + '\n'
		m += 'unusable samples:\t\t' + str(self.unusable_samples) + '\n'
		m += 'perc_unusable:\t\t' + str(self.perc_unusable) + '\n'
		m += 'eeg filename:\t\t' + self.raw_filename + '\n'
		m += 'has veog:\t\t' + str(hasattr(self,'veog'))
		return m


	def __repr__(self):
		if hasattr(self,'name') and hasattr(self,'nblinks'):
			return 'blink-object: '+ self.name+ '\t\tn-blinks: ' + str(self.nblinks)


	def extract_veog(self,b = None):
		if b == None: b = utils.name2block(self.name)
		self.veog_loaded = False
		if not hasattr(b,'raw'): 
			try: 
				b.load_eeg_data() 
				self.ch_index = b.raw.ch_names.index('VEOG')
				print(b.raw.ch_names.index('VEOG'))
			except: return 0 
		'''Extract veog channel from raw mne eeg data object.'''
		# load veog and convert to micro volts ( raw object store values as volts)
		self.veog = b.raw[self.ch_index][0][0] * 10 ** 6
		self.veog_loaded = True


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
		self.start = np.array([p - pre if p > pre else 0 for p in self.peaks])
		# create end point by adding post samples from peak index
		self.end = np.array([p + post if p < len(self.veog) + post else len(self.veog) -1 for p in self.peaks])
		self.nblinks = len(self.peaks)
		self.unusable_samples = (self.pre+self.post) * self.nblinks 
		self.perc_unusable = round((self.unusable_samples / self.n_samples) * 100,2)
		self.validate_peaks()
		# self.prune_peaks()
		self.find_start_end()
		print('found:',self.nblinks,'peaks')

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
		self.calc_unusable()


	def calc_unusable(self):
		self.unusable_samples = (self.pre+self.post) * self.nblinks 
		self.perc_unusable = round((self.unusable_samples / self.n_samples) * 100,2)
	


	def save_blinks(self):
		'''Save blink object with windower make name option.'''
		output = copy.deepcopy(self)
		if hasattr(output,'veog') and self.remove_veog: del output.veog
		if hasattr(output,'model_data'): del output.model_data
		if hasattr(output,'b'): del output.b

		fout = open(path.blinks + self.name + '.blinks','wb')
		pickle.dump(output,fout,-1)
		fout.close()


	def manual_classify_blinks(self):
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

	def write_manual_classification_output(self, cname = ''):
		if cname != '': self.cname = cname
		self.output = []
		for i,p in enumerate(self.peaks):
			l = map(str,[self.st_sample, self.raw_filename, self.marker,p,self.st_sample + p,self.responses[i]])
			self.output.append('\t'.join(l))
		fout = open(path.blinks + self.fn.strip('.blinks') + '_' + str(self.marker) + '_' + self.cname + '.classification','w')
		fout.write('\n'.join(self.output))
		fout.close()

	def plot_peaks(self, raw = None,skip_veog = False):
		'''plot blink epochs and veog channel.'''
		if raw: self.extract_veog(raw)
		t = 'Blinks:'+str(self.nblinks) + '\n\n'
		t +='bad samples:' + str(self.unusable_samples) + '      '
		t +=' perc: ' + str(self.perc_unusable) + '      '
		self.fig = plt.figure()
		self.fig.canvas.set_window_title(self.name)
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

	def plot(self, skip_veog = False):
		if not hasattr(self,'blinks'): self.model_classification()
		'''plot blink epochs and veog channel.'''
		if not hasattr(self,'veog') and not skip_veog: self.extract_veog()
		t = 'Blinks:'+str(len(self.blinks)) + '\n\n'
		self.fig = plt.figure()
		self.fig.canvas.set_window_title(self.name)
		plt.title(t)
		if hasattr(self,'veog') and not skip_veog: plt.plot(self.veog  ,color = 'grey')
		plt.plot(self.blinks,self.value_blinks,'ro')
		plt.plot(self.peaks,self.values,'bo',alpha = 0.2)
		[plt.axvline(s,color='tomato',linestyle='-',linewidth=1) for s in self.start_blinks]
		[plt.axvline(e,color='tomato',linestyle='--',linewidth=1) for e in self.end_blinks]
		plt.legend(('VEOG','Blinks','Peaks'))


	def peaks2model_data(self):
		if not hasattr(self,'veog'): self.extract_veog()
		if not self.veog_loaded: 
			print('could not load veog')
			return 0
		data = []
		self.model_data= np.zeros((len(self.peaks),1000))
		self.model_data[:,:] = np.mean(self.veog)
		for i,p in enumerate(self.peaks):
			if p-500 < 0: 
				start_index = abs(p-500)
				veog_start = 0
			else: 
				start_index = 0
				veog_start = p-500
			if p+500 > self.n_samples: 
				end_index = 1000 - (p+500 - self.n_samples)
				veog_end = self.n_samples 
			else:
				end_index = 1000
				veog_end = p +500
			self.model_data[i,start_index:end_index] = self.veog[veog_start:veog_end]

	def model_classification(self,model_name = '',save = False):
		if not hasattr(self,'model_data'): self.peaks2model_data()
		if not self.veog_loaded: 
			print(self.name,'no model classification')
			return 0
		if model_name == '': model_name = path.data + 'blink-model'
		self.model_name = model_name.split('/')[-1]
		m = blink_model.load_model(model_name = model_name)
		self.correct_peak_indices = np.where(m.prediction_class(self.model_data) == 1)[0]
		self.blinks = self.peaks[self.correct_peak_indices]
		self.value_blinks = self.values[self.correct_peak_indices]
		self.start_blinks = np.array(self.start)[self.correct_peak_indices]
		self.end_blinks = np.array(self.end)[self.correct_peak_indices]
		self.nblinks = len(self.blinks)
		self.calc_unusable()
		m.clean_up()
		print('found:',self.nblinks,'blinks')
		if save: self.write_model_classification()
		return 1
	
	def write_model_classification(self):
		self.output = []
		for i,b in enumerate(self.blinks):
			l = map(str,[self.st_sample, b,self.st_sample + b])
			self.output.append('\t'.join(l))
		fout = open(path.blinks + self.name + '_' + self.model_name+ '.classification','w')
		fout.write('\n'.join(self.output))
		fout.close()




def load_blinks(block):
	'''Load object to file with name == eeg file and extension .blinks in path.blinks folder.'''
	name = windower.make_name(block)
	fn = path.blinks + name + '.blinks'
	if not os.path.isfile(fn):
		print('File does not excist, please provide raw eeg data object',fn)
		return 0
	else: print('loading blinks:',fn)
	fin = open(fn,'rb')
	return pickle.load(fin)


'''

	def load_blinks(self):
		fn = path.blinks + self.vmrk.vmrk_fn.split('/')[-1].strip('.vmrk') + self.exp_type + '_' + str(self.marker) + '.blinks'
		print('looking for:',fn)
		if not os.path.isfile(fn):
			print('File does not excist, loading eeg data')
			self.load_eeg_data()
			self.make_blinks()
		else: self.blinks = blinks.load_blinks(self) 


	def make_blinks(self,remove_veog = True):
		# Make blink object.
		if not self.eeg_loaded:
			self.load_eeg_data()
		if self.raw != 0:
			self.blinks = preprocessing.detect_blinks(self.raw,marker=self.marker,remove_veog = remove_veog)
		else:
			print('cannot make blinks because raw could not be loaded')


'''
