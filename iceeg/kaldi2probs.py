import glob
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import path
import progressbar as pb
from scipy import stats
import time
import multiprocessing


'''
Create pronprob files for each word gate combination. Pronprob files describe the 
probability distribution for phoneme string given a certain gate from word onset.
The pronprob files are made with make_all_pronprobs()
class gates was used to investigate the correct alpha factor (to create probabilities from kaldi output
class kaldi is used to transform nbest into pronprob files for individual words
'''


def get_nbest_files(nbest_dir = None):
	'''files are related to audio file for a block in the eeg-experiment. 
	For each file multiple versions exist, one for each gate. 
	nbest files contain phoneme probabilities for the top 500 phoneme strings 
	for each word in the audio file.
	'''
	if nbest_dir == None: nbest_dir = get_nbest_dir()
	return glob.glob(nbest_dir + '*')

def wav2nbest_files(name, nbest_dir = None):
	'''wav filename has a direct relation to the nbest filename: wavfn_gate_name.best'''
	if nbest_dir == None: nbest_dir = get_nbest_dir()
	fn = get_nbest_files(nbest_dir)
	return [f for f in fn if name in f]

def get_nbest_dir():
	return path.bak + 'NBEST/'

def open_file(f):
	t = [line.split('\t') for line in open(f).read().split('\n') if line]
	chunk_names = list(set([line[0].split('-')[0] for line in t]))
	return t, chunk_names


def get_nbest(t,chunk_name):
	return [line for line in t if chunk_name in line[0]]

class gates():
	'''
	Used to test the best alpha factor to transform kaldi output in to probabilities.
	object to load all nbest list for each gate of the wav filename
	the gates are the number of miliseconds from word onset that is provided to kaldi
	each gate is separately loaded in a kaldi object and consists of a number of chunks
	chunks are the words in the wav file, gated to the n miliseconds of the gate
	for each chunk there is an nbest list of 500 phoneme strings with logprob values.
	the logprob values are not normalized, the alpha is a paramater to normalize the logprobs.
	'''
	def __init__(self,wav_name, alpha = 0.15,alphas = [], nbest_dir = None, compute_all_now = False):
		''' create a gate object with kaldi object for each gate with chunk objects for each word
		in the wav file, gate is a slice of audio from word onset.
		wav_name 		the filename of an audio file
		alpha 			paramater to normalize the kaldi logprob output
		alphas 			series of alpha values to investigate a good alpha values to normalize
						logprob kaldi output
		nbest_dir 		directory of nbest lists
		compute_all... 	whether to compute p and entropy for all gates and alphas
						only needed to investigate a good alpha value in combination with plot method
		'''
		 
		if nbest_dir == None: self.nbest_dir = get_nbest_dir()
		else: self.nbest_dir = nbest_dir
		if alphas == []: self.alphas = np.arange(0.001,0.4,0.005)
		else: alphas = self.alphas
		self.alpha = 0.15
		self.wav_name = wav_name
		self.nbest_filenames = wav2nbest_files(wav_name,self.nbest_dir)
		print('processing:',' '.join(self.nbest_filenames))
		self.gates = [f.split('_')[-1].split('.')[0] for f in self.nbest_filenames]
		self.nbest_files = [kaldi(f) for f in self.nbest_filenames]
		self.entropy = {}
		self.chunk_index = 0
		self.nchunks = len(self.nbest_files[0].chunks)
		if compute_all_now: self.compute_all_alphas()
		self.set_alpha(self.alpha)


	def set_alpha(self,alpha):
		'''sets alpha for all words in all kaldi objects (different gates).'''
		for i,k in enumerate(self.nbest_files):
			k.set_alpha(alpha)


	def compute_all_alphas(self):
		'''compute the entropy of the phon probs for each gate alpha and chunk.
		chunk, an nbest list in kaldi output file with multiple nbest list for each chunk
		'''
		self.chunk_index = -1
		print('computing entropy for all gates alphas and chunks')
		bar = pb.ProgressBar()
		bar(range(self.nchunks))
		while self.next():
			bar.update(self.chunk_index)
			if self.chunk_index >= self.nchunks: break
			self.compute_entropy_all_alphas()

	def compute_entropy_all_alphas(self):
		'''compute the entropy for each gate and alpha.
		gate, is the duration from offset, different gate versions are create for each wav
		alpha, the scaling factor to transform kaldi logprob output to p.
		entropy is computed of the phon pdf
		'''
		temp = np.zeros((len(self.gates),len(self.alphas)))
		for i,k in enumerate(self.nbest_files):
			gate = self.gates[i]
			for j,a in enumerate(self.alphas):
				c = k.chunks[self.chunk_index]
				c.compute_probs(a)
				temp[i,j] = stats.entropy(c.p) 
		self.entropy[self.chunk_index] = temp

				
	def next(self):
		'''handle next chunk, return false if all chunks are processed.'''
		if self.chunk_index < self.nchunks:
			self.chunk_index += 1
			return True
		else: 
			self.chunk_index = 0
			print('done, chunk index reset.')
			return False
		
	
	def plot(self, max_nplots = 10):
		'''plot the entropy accross gates and alphas for multiple chunks.'''
		for chunk_index in self.entropy.keys():
			if int(chunk_index) > max_nplots: break
			name = self.nbest_files[0].chunk_names[chunk_index]
			plt.figure()
			plt.title(name)
			e = self.entropy[chunk_index]
			[plt.plot(self.alphas,e[i], label=gname) for i,gname in enumerate(self.gates)]
			plt.legend()
			plt.grid()

	def _plot_gate(self,gate,e,figure = None, alpha = 0.3,cmap_name = 'tab10',color_scale = 1):
		'''plot the entropy vs alpha for a single gate on a specific figure.
		
		gate 		the length of the chunk from onset
		e 			the entropy scores for this chunk (index for gate)
		figure 		plt figure
		alpha 		transparancy
		'''
		gate_index = self.gates.index(gate)
		cmap = plt.get_cmap(cmap_name)
		color = cmap(gate_index*color_scale,alpha)
		plt.figure(figure.number)
		plt.plot(self.alphas,e[gate_index], color =color)

	def plot_chunks(self, alpha = 0.9,cmap_name = 'coolwarm',color_scale =10,exclude_to_short=True):
		'''plot all chunks in one plot.
		alpha 		plot transparency
		cmap_name 	color mape to be used for plotting
		color_scale how fast color is changing, a good value depends on the number of gates
					40 works for gate 110-250, 10 works for 110-650
		'''
		fig = plt.figure()
		lines, excluded = 0, 0
		for chunk_index in self.entropy.keys():
			for gate in self.gates:
				if exclude_to_short:
					if int(gate) > self.nbest_files[0].chunks[chunk_index].word_dur: 
						excluded += 1
						continue
				e = self.entropy[chunk_index]
				self._plot_gate(gate,e,fig,alpha,cmap_name,color_scale)
				lines += 1
		cmap = plt.get_cmap(cmap_name)
		handles = [patches.Patch(color = cmap(i*color_scale),label=gate) for i,gate in enumerate(self.gates)]
		plt.legend(handles = handles)
		plt.grid()
		print('plotted: ',lines,'excluded:',excluded,'because word is shorter than gate')


def make_all_pronprobs(nprocess= 9,select_subset = ''):
	'''
	Make individual files for each word / gate combination.
	'''
	fn = glob.glob(path.nbest+select_subset+'*.nbest')
	print('processing:',fn,len(fn),'files')
	p = multiprocessing.Pool(nprocess)
	result = p.map(start_kaldi,fn)


def start_kaldi(filename):
	'''helper function to create word_gate pronprob files'''
	print(filename)
	kaldi(filename=filename,save=True,progress = False)
	return time.time()

		
			
	
class kaldi():
	'''process kaldi output to nbest lists of chunks.'''
	def __init__(self,filename, max_nchunks = None, max_nbest_length = 500, alpha = 0.15, save = False, progress=True):
		'''process kaldi output to nbest lists of chunks.

		filename 			filename of kaldi output
		max_nchunks 		speed up processing for testing purposes, default is all
		max_nbest_length 	kaldi output
		'''
		self.filename = filename
		self.gate = filename.split('_')[-1].split('.')[0]
		self.max_nchunks = max_nchunks
		self.max_nbest_length = max_nbest_length
		self.alpha = alpha
		self.experiment_lexicon = get_experiment_lexicon()
		self.show_progress = progress
		self.make_chunks()
		self.get_norms()
		self.nchunks = len(self.chunks)
		self.mean_norm = np.mean(np.array(self.norms))
		self.std_norm = np.std(np.array(self.norms))
		if save: self.save_chunks()

	def __repr__(self):
		m = 'kaldi\t'+self.filename + '\tnchunks: ' + str(self.nchunks) + '\tavg_norm: ' 
		m += str(int(self.mean_norm)) + '\tstd_nrom: '+ str(int(self.std_norm))
		return m


	def __str__(self):
		return self.__repr__()
	

	def set_alpha(self,alpha):
		self.alpha = alpha
		for chunk in self.chunks:
			chunk.compute_probs(alpha)

	
	def make_chunks(self):
		'''make chunk object for each chunk processed by kaldi with nbest list of phonemes strings.'''
		self.chunks = []
		l = self.experiment_lexicon
		print('loading data...')
		t, chunk_names = open_file(self.filename)
		self.chunk_names = chunk_names
		print('processing nbest list for each chunk.''')
		bar = pb.ProgressBar()
		bar(range(len(self.chunk_names)))
		for i,chunk_name in enumerate(self.chunk_names):
			if self.show_progress:bar.update(i)
			if self.max_nchunks and i >= self.max_nchunks:break
			self.chunks.append(chunk(t,chunk_name,self.max_nbest_length,self.alpha,l,self.gate))

	def save_chunks(self):
		for chunk in self.chunks:
			chunk.save()


	def get_norms(self):
		'''get the normalisation value for each chunk 
		(to make sure the values of the nbest list sum to 1)
		'''
		self.norms = [chunk.norm for chunk in self.chunks]


class chunk():
	'''Holds phoneme nbest list corresponding to a chunk of the wav file.'''
	def __init__(self,t,chunk_name, max_nbest_length =None, alpha = 0.15, l = None,gate = ''):
		'''Holds phoneme nbest list corresponding to a chunk of the wav file.

		t 			text file containing kaldi output
		chunk_name 	the name of the current chunk
		'''
		self.chunk_name = chunk_name.split(' ')[0]
		self.name = chunk_name.split('.')[0]
		self.nbest = get_nbest(t,chunk_name)
		self.experiment_lexicon = l
		self.gate = gate
		if max_nbest_length and max_nbest_length < len(self.nbest):
			self.nbest = self.nbest[:max_nbest_length]
		self.compute_probs(alpha)
		self.get_word()
		

	def __repr__(self):
		m = 'chunk\t'+ self.chunk_name + '\tnbest_length: ' + str(len(self.nbest))
		m += '\tnorm: ' + str(int(self.norm))
		return m

	def __str__(self):
		return __repr__(self)

	def get_word(self):
		if self.experiment_lexicon == None: self.experiment_lexicon = get_experiment_lexicon()
		# print(self.chunk_name,self.chunk_name.split('.'))
		self.word_id = int(self.chunk_name.split('.')[-1].lstrip('0'))
		self.word,self.wid_check,self.word_phon,self.word_dur=self.experiment_lexicon[self.word_id-1]
		assert self.word_id == int(self.wid_check)
		self.word_dur = int(self.word_dur)
		
		
	def compute_probs(self, alpha = 0.15):
		self.alpha = alpha
		self.score = np.array([float(line[1]) for line in self.nbest])
		self.phons = [line[0].split(' ')[1] for line in self.nbest]
		temp = np.exp(alpha * (min(self.score) - self.score))
		self.norm = sum(temp)
		self.p = temp / self.norm
		self.logprob = np.log10(self.p)

	def save(self, make_gate_dir = True):
		if make_gate_dir:
			d = path.pronprob + self.gate
			if not os.path.isdir(d): os.mkdir(d)
			gd = self.gate +'/'
		else: gd = ''
		salpha = str(self.alpha).replace('.','')
		with open(path.pronprob+gd+'word_'+self.wid_check + '_' + self.gate+'_'+salpha+'.pronprob','w') as fout: 
			o='\n'.join(['\t'.join(map(str,line)) for line in zip(self.phons,self.p)])
			fout.write(o)
			


def get_experiment_lexicon():
	temp = open(path.data +'experiment_lexicon_w2i-pron-dur.txt').read().split('\n')
	return [line.split('\t') for line in temp]


'''
def nbest2probs(nbest):
	score = [-1 * line[1] for line in nbest]
	phons = [line[-1] for line in nbest]
	temp = score -min(score)
	norm = sum(temp)
	p = temp / norm
	log_prob = np.log10(p)
	phon_prob = dict(zip(phons,p))
	phon_logprob = dict(zip(phons,logprob))
	return phon_prob, phon_logprob, norm


"k = kp.kaldi('/Users/u050158/fn001161_110.total',max_nchunks=100)",
 'k.chunks',
 'plt.clf()',
 'plt.plot(k.norms)',
 "get_ipython().run_line_magic('pinfo', 'plt.ylim')",
 'plt.ylim(0,90000)',
 'plt.grid()',
 'k.mean_norm',
 'k.std_norm',
 'plt.clf()',
 '[plt.plot(c.p) for c in k.chunks,alpha=.2]',
 '[plt.plot(c.p,alpha = 0.2) for c in k.chunks]',
 'plt.clf()',
 "[plt.plot(c.p,alpha = 0.1,color = 'black') for c in k.chunks]",
 'plt.grid()',
 'c = k.chunks[0]',
 'c',
 'max(c.score)',
 'min(c.score)',
 'mins = [min(c.score) for c in k.chunks]',
 'maxs = [max(c.score) for c in k.chunks]',
 'plt.clf()',
 "plt.plot(mins,color='red')",
 "plt.plot(maxs,color='blue')",
 'mean = [np.mean(c.score) for c in k.chunks]',
 "plt.plot(mean,color='purple')",
 "plt.legend('min','max','avg')",
 "get_ipython().run_line_magic('pinfo', 'plt.legend')",
 'plt.clf()',
 "plt.plot(mean,color='purple',label='avg')",
 "plt.plot(maxs,color='blue',label='max')",
 "plt.plot(mins,color='red',label='min')",
 'plt.grid()',
 'plt.legend()',
 'plt.clf()',
 "[plt.plot(c.score,alpha = 0.1,color = 'black') for c in k.chunks]",
 'plt.grid()',

'''

