import numpy as np
import progressbar as pb

def open_file(f):
	t = [line.split('\t') for line in open(f).read().split('\n') if line]
	chunk_names = list(set([line[0].split('-')[0] for line in t]))
	return t, chunk_names


def get_nbest(t,chunk_name):
	return [line for line in t if chunk_name in line[0]]
	
	
class kaldi():
	'''Process kaldi output to nbest lists of chunks.'''
	def __init__(self,filename, max_nchunks = None, max_nbest_length = 5000):
		'''Process kaldi output to nbest lists of chunks.

		filename 		filename of kaldi output
		'''
		self.filename = filename
		self.max_nchunks = max_nchunks
		self.max_nbest_length = max_nbest_length
		self.make_chunks()
		self.get_norms()
		self.nchunks = len(self.chunks)
		self.mean_norm = np.mean(np.array(self.norms))
		self.std_norm = np.std(np.array(self.norms))

	def __repr__(self):
		m = 'kaldi\t'+self.filename + '\tnchunks: ' + str(self.nchunks) + '\tavg_norm: ' 
		m += str(int(self.mean_norm)) + '\tstd_nrom: '+ str(int(self.std_norm))
		return m


	def __str__(self):
		return self.__repr__()
	
	
	def make_chunks(self):
		'''Make chunk object for each chunk processed by kaldi with nbest list of phonemes strings.'''
		self.chunks = []
		print('loading data...')
		t, chunk_names = open_file(self.filename)
		self.chunk_names = chunk_names
		print('processing nbest list for each chunk.''')
		bar = pb.ProgressBar()
		bar(range(len(self.chunk_names)))
		for i,chunk_name in enumerate(self.chunk_names):
			bar.update(i)
			if self.max_nchunks and i >= self.max_nchunks:break
			self.chunks.append(chunk(t,chunk_name,self.max_nbest_length))


	def get_norms(self):
		'''Get the normalisation value for each chunk 
		(to make sure the values of the nbest list sum to 1)
		'''
		self.norms = [chunk.norm for chunk in self.chunks]


class chunk():
	'''Holds phoneme nbest list corresponding to a chunk of the wav file.'''
	def __init__(self,t,chunk_name, max_nbest_length =None):
		'''Holds phoneme nbest list corresponding to a chunk of the wav file.

		t 			text file containing kaldi output
		chunk_name 	the name of the current chunk
		'''
		self.chunk_name = chunk_name.split(' ')[0]
		self.name = chunk_name.split('.')[0]
		self.nbest = get_nbest(t,chunk_name)
		if max_nbest_length and max_nbest_length < len(self.nbest):
			self.nbest = self.nbest[:max_nbest_length]
		self.compute_probs()
		

	def __repr__(self):
		m = 'chunk\t'+ self.chunk_name + '\tnbest_length: ' + str(len(self.nbest))
		m += '\tnorm: ' + str(int(self.norm))
		return m

	def __str__(self):
		return __repr__(self)
		
		
	def compute_probs(self):
		self.score = np.array([-1 * float(line[1]) for line in self.nbest])
		self.phons = [line[0].split(' ')[1] for line in self.nbest]
		temp = 2**(self.score -min(self.score))
		self.norm = sum(temp)
		self.p = temp / self.norm
		self.logprob = np.log10(self.p)

	def save(self):
		with open(self.chunk_name) as fout: 
			o='\n'.join(['\t'.join(map(str,line)) for line in zip(self.score,self.phons,self.p,self.logprob)])
			fout.write(o)
			



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

