import copy
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

class Garbage_collector:
	def __init__(self,session,length = 1,remove_ch = None):
		self.session = session
		self.length = length
		self.remove_ch = remove_ch
		self.gather_block_statistics()
		self.make_blockstats_available()
		self.set_channel_garbage()
		self.set_epoch_garbage()
		for bs in self.bstats: bs.aggregate_garbage()
		

	def gather_block_statistics(self):
		self.bstats = []
		for b in self.session.blocks:
			self.bstats.append(Garbage_stats(b,self.length,self.remove_ch))


	def make_blockstats_available(self):
		'''make block stats available on object as a property .b1 .b2 .b3 etc.'''
		self.bsnames= ['b' +str(i) for i in range(1,len(self.bstats)+ 1)]
		[setattr(self,bname,self.bstats[i]) for i,bname in enumerate(self.bsnames)]


	def set_channel_garbage(self):
		self.l_chvariance = [bs.channel_variance for bs in self.bstats]
		self.l_chcorrelation = [bs.channel_correlation for bs in self.bstats]

		self.np_chvariance = np.concatenate(self.l_chvariance)
		self.np_chcorrelation = np.concatenate(self.l_chcorrelation)
		
		for n in ['np_chvariance','np_chcorrelation']: self.calc_zscore(n)


	def set_epoch_garbage(self):
		self.l_epvariance = [bs.epoch_variance for bs in self.bstats]
		self.l_epdeviation = [bs.epoch_deviation for bs in self.bstats]
		self.l_epamprange = [bs.epoch_amprange for bs in self.bstats]

		self.np_epvariance = np.concatenate(self.l_epvariance)
		self.np_epdeviation = np.concatenate(self.l_epdeviation)
		self.np_epamprange = np.concatenate(self.l_epamprange)

		names = ['np_epvariance','np_epdeviation','np_epamprange']
		for n in names: self.calc_zscore(n)


	def calc_zscore(self,input_attr):
		zname = input_attr + '_zscore'
		setattr(self,zname, stats.zscore(getattr(self,input_attr)))
		self.set_zscore(input_attr)


	def set_zscore(self,name):
		zname = name + '_zscore'
		bzname = zname.lstrip('np_')
		all_name = 'l_' + name.lstrip('np_')
		s_i_bstats, e_i_bstats, t = [],[],0 
		for item in getattr(self,all_name):
			s_i_bstats.append(t)
			t+= len(item)
			e_i_bstats.append(t)
		for i,bs in enumerate(self.bstats):
			setattr(bs,bzname,getattr(self,zname)[s_i_bstats[i]:e_i_bstats[i]])
			bs.threshold_zscore(name)
		




class Garbage_stats:
	def __init__(self,block,length = 1,remove_ch= None):
		self.block = block
		self.length = int(float(length) * 1000)
		if remove_ch != None and type(remove_ch) == list: self.remove_ch = remove_ch
		else: self.remove_ch = ['VEOG','HEOG','TP10_RM','STI 014','LM']
		self.load_data()
		self.remove_channels()
		self.set_info()
		self.set_garbage()

	def load_data(self):
		if not hasattr(self.block,'raw'):
			self.block.load_eeg_data()
		self.ch_names = self.block.raw.ch_names
		self.data = self.block.raw[:][0] * 10 ** 6
		self.st_sample = self.block.raw.first_samp
		self.duration_sample = len(self.block.raw)
		self.raw_ch_names = self.block.raw.ch_names
		del self.block.raw

	def remove_channels(self,channels = []):
		self.remove_ch += channels
		self.ch_mask = [n not in self.remove_ch for n in self.ch_names]
		self.ch_names= [n for n in self.raw_ch_names if not n in self.remove_ch]
		self.data = self.data[self.ch_mask,:]
		

	def set_info(self):
		self.marker = self.block.marker
		self.start_snippets = np.arange(0,self.duration_sample,self.length)
		self.end_snippets = self.start_snippets + self.length
		# last snippet can only last until end data
		self.end_snippets[-1] = self.duration_sample
		if self.end_snippets[-1] - self.start_snippets[-1] < 500:
			# minimum snippet length = 500, maximum = 1499
			self.start_snippets = np.delete(self.start_snippets,-1)
			self.end_snippets = np.delete(self.end_snippets,-2)
		self.mean_ch = self.data.mean(1)

	def set_garbage(self):
		self.calc_channel_variance()
		self.calc_channel_correlation()

		self.calc_epoch_amprange()
		self.calc_epoch_deviation()
		self.calc_epoch_variance()

	def calc_channel_variance(self):
		self.channel_variance = self.data.var(1)

	def calc_channel_correlation(self):
		self.corr_matrix = np.corrcoef(self.data)
		self.channel_correlation = self.corr_matrix.mean(1)


	def calc_epoch_variance(self):
		'''Calculate mean across channels of variance for each snippet.
		'''
		self.epoch_variance = np.zeros(len(self.start_snippets))
		for i,(start,end) in enumerate(zip(self.start_snippets,self.end_snippets)):
			#variance in row direction (1), mean in column direction, accross channels
			self.epoch_variance[i] = self.data[:,start:end].var(1).mean()


	def calc_epoch_deviation(self):
		'''Calculate mean across channels of deviation for each snippet.
		'''
		self.epoch_deviation= np.zeros(len(self.start_snippets))
		for i,(start,end) in enumerate(zip(self.start_snippets,self.end_snippets)):
			# mean in row direction (1) minus channel mean of block
			dev = self.data[:,start:end].mean(1) - self.mean_ch
			# mean in column direction, accross channels
			self.epoch_deviation[i] = dev.mean()
	

	def calc_epoch_amprange(self):
		'''Calculate mean across channels of deviation for each snippet.
		'''
		self.epoch_amprange = np.zeros(len(self.start_snippets))
		for i,(start,end) in enumerate(zip(self.start_snippets,self.end_snippets)):
			# max minus min of each epoch, mean accross channels for each epoch
			amprange =  self.data[:,start:end].max(1) -  self.data[:,start:end].min(1) 
			self.epoch_amprange[i] = amprange.mean() 


	def threshold_zscore(self,input_attr):
		input_attr = input_attr.lstrip('np_')
		zscore,bi,names = [input_attr + n for n in  ['_zscore','_bi','_names']]
		setattr(self,bi, np.where(abs(getattr(self,zscore))> 3)[0])
		if '_ch' in input_attr:
			setattr(self,names, [n for i,n in enumerate(self.ch_names) if i in getattr(self,bi)])

	def aggregate_garbage(self):
		self.epoch_all= [self.epamprange_bi,self.epdeviation_bi,self.epvariance_bi]
		self.epoch_bi = np.concatenate(self.epoch_all)
		self.epoch_bi = np.array(list(set(self.epoch_bi)))

		self.channel_all = [self.chcorrelation_bi,self.chvariance_bi]
		self.channel_bi = np.concatenate(self.channel_all)
		self.bad_ch_names= [n for i,n in enumerate(self.ch_names) if i in self.channel_bi]

	def remove_bad_channels(self):
		self.remove_channels(self.bad_ch_names)


	def plot_bad_epoch(self,channels = ['Fz','Cz','Pz'],dec = 10):
		if channels == []: channels = 'all'
		if channels == 'all': channels = self.ch_names
		if type(channels[0]) == int: 
			channel_index= copy.deepcopy(channels)
			channels = [g.ch_names[i] for i in channels]
		else: channel_index = [self.ch_names.index(n) for n in channels]
		plt.figure(str(self.marker))

		start = self.start_snippets[self.epoch_bi]
		end = self.end_snippets[self.epoch_bi]

		# data is lowpass filtered at 30 Hz, nyquist = 60 Hz and sf = 1000, so
		# decimate should not exceed 15, 10 speeds up plotting nicely
		if dec > 15: dec = 15 
		yx = np.arange(0,self.data.shape[1],dec)
		y = self.data[:,yx]
			
		offset = 0
		for i in channel_index: 
			plt.plot(yx,y[i,:]+offset,linewidth = 0.5)
			offset += 40
		plt.ylim((-40,(len(channel_index) + 2) * 40))

		[plt.axvline(s,color='tomato',linestyle='-',linewidth=1,alpha=0.1) for s in start]
		[plt.axvline(e,color='tomato',linestyle='--',linewidth=1,alpha=0.1) for e in end]
		plt.legend(channels)

