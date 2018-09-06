from matplotlib import pyplot as plt
import numpy as np
import path
import utils


class infos:
	'''create object with data that describe the performance of channel cnn model.'''
	def __init__(self,identifier= ''):
		'''The info is read based on the identifier.'''
		self.data = read_data(identifier =identifier) 
		self.infos = []
		for line in self.data:
			self.infos.append(classify_info(line))
		self.get_stats()
		self.infos_dict = {}
		for info in self.infos:
			self.infos_dict[info.name] = info

	def __repr__(self):
		m = 'infos ch object\t'
		m += 'avg mcc:  ' + str(round(self.avg_mcc,3)) + '\t'
		m += 'avg precision:  ' + str(round(self.avg_precision,3)) + '\t'
		m += 'avg recall:  ' + str(round(self.avg_recall,3)) + '\t'
		return m

	def get_stats(self):
		'''Reads in the statistics per channel for each manually annotated block, to test
		performance of the cnn model.
		'''
		self.mcc, self.precision, self.recall = [], [] ,[] 
		self.tn, self.fp, self.fn,self.tp = [0]*4
		self.err,self.err_fn,self.err_fp = {},{},{}
		for ch in self.infos[0].pc.keys():
			self.err[ch],self.err_fp[ch], self.err_fn[ch] = 0 , 0 , 0
		for i,info in enumerate(self.infos):
			self.mcc.append( info.mcc )
			self.precision.append( info.precision )
			self.recall.append( info.recall )
			self.tn += info.tn
			self.fp += info.fp
			self.fn += info.fn
			self.tp += info.tp
			for ch in info.err.keys():
				self.err[ch] += info.err[ch]
				self.err_fn[ch] += info.err_fn[ch]
				self.err_fp[ch] += info.err_fp[ch]
		self.cm = np.array(([self.tn,self.fp],[self.fn,self.tp]))
		self.norm_cm = (self.cm/(self.tn/10000)).astype(int)
		self.avg_mcc = sum(self.mcc) / len(self.infos)
		self.avg_precision = sum(self.precision)/ len(self.infos)
		self.avg_recall = sum(self.recall) / len(self.infos)
		self.n_samples = self.tn + self.fp + self.fn + self.tp
		self.classified_artifacts = self.tp + self.fp
		self.classified_clean = self.tn + self.fn
		self.artifacts = self.tp + self.fn
		self.clean = self.tn + self.fp
		self.errv2k = dict([[self.err[ch],ch] for ch in self.err.keys()])

	def plot_mistakes_per_channel(self):
		'''Show the total number of miss clasified indices of the cnn annotation, compared to
		the manual annotation.
		'''
		self.chs = []
		self.sorted_mistakes = sorted(self.errv2k.keys())[::-1]
		
		for k in self.sorted_mistakes:
			self.chs.append(self.errv2k[k])
		fn = [self.err_fn[ch] for ch in self.chs]
		fp = [self.err_fp[ch] for ch in self.chs]
		plt.figure()
		plt.plot(self.sorted_mistakes,'ro')
		plt.plot(fn,'bo')
		plt.plot(fp,'go')
		plt.legend(('wrong','false negative','false positive'))
		plt.plot(self.sorted_mistakes)
		plt.plot(fn,'grey')
		plt.plot(fp,'grey')
		[plt.annotate(self.chs[i], xy=(i,self.sorted_mistakes[i])) for i in range(len(self.sorted_mistakes))]
		plt.grid()
		

class classify_info:
	'''Create different classification performance stats based on the info file.'''
	def __init__(self,line):
		self.line = line
		self.channels = utils.load_selection_ch_names()
		# [f,p.mcc,p.precision,p.recall,p.cm.tn,p.cm.fp,p.cm.fn,p.cm.tp,gt,pc])))
		self.name = line[0].split('_perc.npy')[0]
		self.mcc = float(line[1])
		self.precision = float(line[2])
		self.tn = int(line[4])
		self.fp = int(line[5])
		self.fn = int(line[6])
		self.tp = int(line[7])
		self.pc = dict(zip(self.channels,list(map(int,line[-1].split(',')))))
		self.gt = dict(zip(self.channels,list(map(int,(line[-2].split(','))))))
		self.err, self.err_fn, self.err_fp = {},{},{}
		for ch in self.pc.keys():
			self.err[ch] = abs(self.gt[ch] - self.pc[ch])
			if self.gt[ch] - self.pc[ch] < 0:
				self.err_fp[ch] = self.err[ch]
				self.err_fn[ch] = 0
			else:
				self.err_fn[ch] = self.err[ch]
				self.err_fp[ch] = 0
		self.n_samples = self.tn + self.fp + self.fn + self.tp

		for ch in self.pc.keys():
			pass
			# dif = self.pc[ch] - self.gt
			
		if self.tp == 0 and self.fp != 0: self.recall = 0
		else: self.recall = float(line[3])





def read_data(name = 'classify_ch_info',identifier = ''):
	''' read in file with statistics about cnn classified ch data, to check
	how well the cnn performed on ch data.
	file is made with cnn_ch_output_data
	name 		filename, 
				other options classify_ch_info_co100, classify_ch_info_co100_ch26
	'''
	if len(identifier) > 0 and identifier[0] != '_': identifier = '_' + identifier
	return [line.split('\t') for line in open(path.data +name+identifier).read().split('\n') if line]


