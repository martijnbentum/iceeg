import glob
import kaldi2probs
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os
import path
import pmn_input_data as pid
import progressbar as pb
import random
from scipy import stats
import utils

'''
Create cross entropy, entropy, updated entropy, surprisal, updated surprisal
for all word (50k) and gate (110, 130, ..., 650) combinations in the experiment.

class dataset_pdf creates a single dataset for a gate, this was to slow to be workable.
dataset_pdf is not used. To create pmn_datawords do the following:

1) load pmn_words with:
d = pp.dataset_pdf(gate=110,overwrite=False)
pmn_words = d.pmn_words

pmn_words are made with pid.pmn_data class

2) To create pmn_datawords use:
multi_pmn(pmn_words = pmn_words,gate = gate,nthreads = 6, test = False):

pmn_dataword contains the abovementioned information theoretic measures
they are stored per word in a directory of the corresponding gate

the measures are created by generating a word probability distribution based on a SLM
a phoneme string probability distribution (phoneme strings can be 1-8 in length) based on Kaldi
the wpd is updated based on the ppd, the cross entropy is computed on the wpd and updated wpd
updated entropy and surprisal is based on the udpated wpd
'''

#frequency of a word in the COW corpus
wtd = utils.load_dict_wordtype2freq() 

class dataset_pdf():
	'''Creates a dataset of pmn_datawords, is to slow and therefore obsolete
	use multi_pmn
	'''
	def __init__(self,gate, n_pronprob = 50,filename = '', pmn_words = None,overwrite = True):
		if pmn_words == None: 
			pmn = pid.pmn_data(history_length=3)
			self.pmn_words = pmn.pmn_words
		else: self.pmn_words = pmn_words
		self.gate = str(gate)
		self.n_pronprob = n_pronprob
		self.made = False
		if filename == '': 
			self.filename = path.pmn_datasets+'pmn_dataset_g-'+str(gate) +'_pp-'+str(n_pronprob)
		else: self.filename = filename
		self.filename_temp = self.filename + '_temp'
		if overwrite: open(self.filename_temp,'w').close()
		self.kaldi_dict = make_kaldi_dict()
		self.output,self.failed,self.bad_word = [], [], []


	def check_word(self,pmn_word):
		w = pmn_word.word
		if not hasattr(w,'ppl') or not hasattr(w,'pos'): return False
		if w.word_utf8_nocode_nodia().lower() not in wtd.keys(): return False
		else: return True


	def make_dataset(self,save = False):
		bar = pb.ProgressBar()
		bar(range(len(self.pmn_words)))
		for i,pmn_word in enumerate(self.pmn_words):
			bar.update(i)
			if self.check_word(pmn_word):
				pmndw = pmn_dataword(pmn_word,self.kaldi_dict,self.gate,self.n_pronprob)
				if pmndw.ok: 
					self.output.append(pmndw.output_str)
					self.save_line(pmndw.output_str)
				else: self.failed.append(pmn_word)
			else:self.bad_word.append(pmn_word)
		self.header = '\t'.join(pmndw.header())
		self.made = True
		self.output.insert(0,self.header)
		self.save()


	def save(self):
		fout = open(self.filename,'w')
		fout.write('\n'.join(self.output))
		fout.close()


	def save_line(self, line):
		fout = open(self.filename_temp,'a')
		fout.write(line + '\n')
		fout.close()
		

def multi_pmn(pmn_words = [],gate = '210',nthreads = 6, test = False):
	'''Multi-threaded function to create pmn_datawords.
	Each word is saved seperately (this prevends threading isues with writing to file)
	'''
	if pmn_words == []:
		pmn = pid.pmn_data(history_length=3)
		pmn_words = pmn.pmn_words
	p = multiprocessing.Pool(nthreads)
	if not os.path.isdir(path.pmn_datasets + gate): 
		print('making new dirs for gate',gate)
		os.mkdir(path.pmn_datasets + gate)
		os.mkdir(path.pmn_datasets + gate + '/WORDS')
		os.mkdir(path.pmn_datasets + gate + '/BAD_WORDS')
		os.mkdir(path.pmn_datasets + gate + '/FAILED')
	for pmnw in pmn_words:
		pmnw.gate = gate
	if test:
		result = p.map(make_pmn_dataword,pmn_words[:30])
	else:
		result = p.map(make_pmn_dataword,pmn_words)

def make_pmn_dataword(pmn_word):
	'''compute the information theoretic measures based on SLM ASR output.
	assumes the gate number is added to the pmn_word
	
	first checks whether word is already processed to prevend double work

	the creation of a pmn_dataword can result in the following outcomes
	a word can be bad, if it does not have ppl (surprisal info) or pos (part of speech info)
	a word can fail, if the pmn_dataword is not succesfull
	a word can succeed
	words will be save in BAD_WORDS, FAILED or WORDS directory
	'''
	gate = pmn_word.gate
	w = pmn_word.word
	word = w.word_utf8_nocode_nodia().lower()
	filename = pmn_word.index + '_' + word
	if os.path.isfile(path.pmn_datasets + gate +'/BAD_WORDS/' + filename):
		print('skipping bad word',filename,gate)
		return
	if os.path.isfile(path.pmn_datasets + gate + '/WORDS/' + filename):
		print('skipping good word',filename,gate)
		return
	if os.path.isfile(path.pmn_datasets + gate + '/FAILED/' + filename):
		print('skipping failed word',filename,gate)
		return
	print('new word',filename,gate)
	if not hasattr(w,'ppl') or not hasattr(w,'pos'): 
		open(path.pmn_datasets +gate +'/BAD_WORDS/' + pmn_word.index + '_' + word,'w').close()
		print('bad word',pmn_word.index,word)
		return False
	kd = make_kaldi_dict()
	pdw = pmn_dataword(pmn_word,kd,gate,50)
	if pdw.ok:
		fout = open(path.pmn_datasets +gate+ '/WORDS/'+filename,'w')
		fout.write(pdw.output_str + '\n')
		fout.close()
		print('good word',pmn_word.index,word)
	else:
		open(path.pmn_datasets + gate +'/FAILED/' + filename,'w').close()
		print('failed word',pmn_word.index,word)
		return False

		
class pmn_dataword():
	'''Class to create word probability distribution, phoneme probability distribution
	and compute the update of wpd based on the ppd and information theoretic measures.
	'''
	def __init__(self,pmn_word,kaldi_dict,gate,n_pronprob):
		'''create a pmn_dataword an object with word and phoneme pdfs that can update wpd
		the object contain a word probability distribution based on a slm and a
		phoneme string probability distribution based on asr (kaldi)
		these distributions are created with the pdf class

		pmn_word 		object pmn word info
		kaldi_dict 		maps phoneme string name to phoneme list (phoneme space seperated)
		gate 			the gate number
		n_pron... 		the top n phonemes used from the kaldi output
		'''
		self.pmn_word = pmn_word
		self.w = pmn_word.word
		self.word = self.w.word_utf8_nocode_nodia().lower()
		self.duration = self.w.duration_sample
		self.usable = self.w.usable
		self.content_word= self.w.pos.content_word
		self.kaldi_dict = kaldi_dict
		self.gate = gate
		self.n_pronprob = n_pronprob
		self.make()

	def make(self):
		self.cow,self.pp,self.wnp = pmn2cow(self.pmn_word,kaldi_dict=self.kaldi_dict,gate =self.gate,n_pronprob = self.n_pronprob)
		if not self.cow: self.ok = False
		else: self.ok = True
		if self.ok: self._set_info()
		if self.ok: self.data()
			

	def _set_info(self):
		'''create all info for the pmn_dataword.'''
		self.index = int(self.pmn_word.index)
		self.word_block_index = self.w.ppl.word_index
		self.word_id = self.w.ppl.word_id
		self.word_in_sentence = self.w.word_number + 1
		self.fid = self.w.fid
		self.sid = self.w.sid
		self.cow_fn = self.cow.filename.split('/')[-1]
		self.pp_fn = self.pp.filename.split('/')[-1]
		self.wnp_fn = self.wnp.filename.split('/')[-1]
		self.filenames = ','.join([self.cow_fn,self.pp_fn,self.wnp_fn])
		self.entropy = self.cow.entropy
		self.ud_entropy = self.cow.ud_entropy
		self.cross_entropy = self.cow.cross_entropy
		self.freq = int(wtd[self.word]) + 1
		self.freq_log = np.log(self.freq)
		self.ngram = self.w.ppl.ngram
		self.cow.sort()
		self.cow_top5_ort = ','.join([item.word for item in self.cow.items[:5]])
		self.cow_top5_phon= ','.join(['-'.join(item.phon) for item in self.cow.items[:5]])
		target,self.cow_index_word = self.cow.find_item(self.word)
		self.ppl_surprisal = self.w.ppl.logprob
		if self.cow_index_word == None: self.ok = False
		else:
			self.exp = self.w.ppl.exp
			self.surprisal = target.logprob
			self.ud_surprisal = target.updated_lp
		self.cow.sort(prob_type = 'updated_p')
		self.ud_top5_ort = ','.join([item.word for item in self.cow.items[:5]])
		self.ud_top5_phon= ','.join(['-'.join(item.phon) for item in self.cow.items[:5]])
		_,self.ud_index_word = self.cow.find_item(self.word)
		self.cow.sort(prob_type = 'diff_p')
		self.diff_top5_ort = ','.join([item.word for item in self.cow.items[:5]])
		self.diff_top5_phon= ','.join(['-'.join(item.phon) for item in self.cow.items[:5]])
		_,self.diff_index_word = self.cow.find_item(self.word)

	def header(self):
		'''create a header that explains what each column contains.'''
		m = 'word,index,fid,sid,exp,duration,usable,content_word,gate,n_pronprob'
		m += ',ud_entropy,entropy,cross_entropy,freq,freq_log,ngram,word_in_sentence'
		m += ',surprisal,ud_surprisal,ppl_surprisal,cow_index_word,ud_index_word,diff_index_word'
		m += ',cow_top5_ort,ud_top5_ort,diff_top5_ort,cow_top5_phon,ud_top5_phon'
		m += ',diff_top5_phon,filenames,word_id,word_block_index'
		return m.split(',')

	def data(self):
		'''extract data from object.'''
		header = self.header()
		self.output_line = []
		for h in header:
			self.output_line.append(str(getattr(self,h)))
		self.output_str = '\t'.join(self.output_line)
			
		

class pdf():
	'''object to hold pdf of words or phonemes.'''
	def __init__(self,filename,pdf_type,kaldi_dict=None,onset_dict = None, n_pronprob = 50):
		'''object to hold pdf of words or phonemes.
		filename 		filename of pdf
		pdf_type 		type of pdf (words_cow, words_np, pronprob) see below
		kaldi_dict 		needed for pronprob to set phonemes

		pdf_types:
		words_cow		set of words that have phoneme transcription, 
						pdf that is updated(by pronprob) and set by words_np and the cross entropy is calculated over
		words_np 		np_array of probablities index of values corresponds to index of words_cow
		pronprob 		nbest list of phoneme strings with probs
		'''
		self.filename = filename
		self.pdf_type = pdf_type
		self.kaldi_dict = kaldi_dict
		self.onset_dict = onset_dict
		self.n_pronprob = n_pronprob
		if pdf_type == 'words_np': self.probs= np.load(self.filename)
		else: 
			self.lexicon= open_lexicon(self.filename)
			self.items = []
			for line in self.lexicon:
				if line[1] == 'NA': continue
				self.items.append(lex_word(line,self.pdf_type,kaldi_dict=self.kaldi_dict)) 
		self._set_info()

	def __repr__(self):
		m = 'pdf-'+self.pdf_type  
		if self.pdf_type == 'words_cow': 
			m +=' \tprob_set: '+str(self.probs_set)
			m += ' \tentropy: ' + str(self.entropy)
			m += ' \tcross_entropy: ' + str(self.cross_entropy)
		if self.pdf_type == 'words_np': 
			m +=' \tpreceding-words: '+'-'.join(self.preceding_words)
		if self.pdf_type == 'pronprob': 
			m +=' \tgate: '+str(self.gate)
			m +=' \tshifted-lp: '+str(self.shifted_logprob)
		return m

	def __str__(self):
		m = 'pdf-'+self.pdf_type 
		m += '\npdf_size: \t\t' + str(self.pdf_size)
		m +='\nprob_set: \t\t'+str(self.probs_set)
		m +=' \npreceding-words: \t'+' '.join(self.preceding_words)
		m +=' \ngate: \t\t\t'+str(self.gate)
		m +=' \nshifted-lp: \t\t'+str(self.shifted_logprob)
		m += '\nentropy: \t\t' + str(self.entropy)
		m += '\ncross_entropy: \t\t' + str(self.cross_entropy)
		return m

	def _set_info(self):
		'''set information that tells what kind of pdf this is.'''
		self.preceding_words,self.gate,self.pdf_size = 'na','na','na'
		self.entropy, self.cross_entropy = 'na', 'na'
		self.probs_set,self.shifted_logprob = 'na','na'

		if self.pdf_type == 'words_np': 
			self.preceding_words = self.filename.split('.')[0].split('/')[-1].split('_')
			self.probs_set = True
		if self.pdf_type == 'pronprob': 
			self.gate = int(self.filename.split('_')[-2])
			self.shifted_logprob = False
			self.probs_set = True
		if self.pdf_type == 'words_cow': 
			self.pdf_size = len(self.items)
			self.probs_set = False
			self.all_phons = list(set(' '.join([' '.join(w.phon) for w in self.items]).split(' ')))
			if self.onset_dict == None: self.make_onset_dict()
		# else: self.calc_entropy()
		
	def make_onset_dict(self):
		'''create a dictionary to speed up kaldi search (only search in set with matching onset).'''
		self.onset2word= {}
		for phon in self.all_phons:
			if phon not in self.onset2word.keys(): self.onset2word[phon] =[]
			for word in self.items:
				if word.match_onset(phon): self.onset2word[phon].append(word)


	def find_item(self,label):
		'''find the index of an item in the pdf.'''
		for index,item in enumerate(self.items):
			if item.word == label:
				return item,index
		return 'not_found',None

	def sort(self, reverse = True, prob_type = 'prob'):
		'''sort the pdf based on different measures.'''
		if prob_type == 'prob':
			self.items.sort(reverse = reverse,key =lambda x: x.prob)
		if prob_type == 'diff_p':
			self.items.sort(reverse = reverse,key =lambda x: x.diff_p)
		if prob_type == 'updated_p':
			self.items.sort(reverse = reverse,key =lambda x: x.updated_p)
		# except: print('some items do not have probs, doing nothing')

	def set_pdf(self,pdf):
		if pdf.pdf_type == 'words_np' and self.pdf_type == 'words_cow':
			for item in self.items:
				item.set_prob(logprob = pdf.probs[item.index])
			self.probs_set = True
			self.preceding_words = pdf.preceding_words
			self.calc_entropy()
		else:print(self.pdf_type,'not defined for set_pdf')
			
	def calc_entropy(self):
		'''calculate the entropy of the probability distribution.'''
		if self.pdf_type == 'words_np': self.entropy = stats.entropy(self.probs)
		else: 
			self.entropy = stats.entropy([item.prob for item in self.items])

	def _reset_update(self):
		for item in self.items:
			item.reset_update()
		
		
	def _shift_logprobs(self, shift):
		'''logprobs have the unwanted effect of having the highest value for unlikely items.
		we use only the top n phonemes, we therefore shift the logprob of the phonemes with
		the logprob of the lowest ranking phoneme string, this results in the biggest shift
		of word that match the most likely phoneme strings.
		normalization ensure that the udpated wpd is a valid probability distribution
		'''
		if self.pdf_type != 'pronprob': raise ValueError('can only shift logprobs for pronprob pdf')
		if self.shifted_logprob: raise ValueError('can only shift logprobs once')
		for item in self.items:
			item.logprob += abs(shift)
		self.shifted_logprob = True


	def _calc_p(self):
		'''create a valid probability distribution by normalization.'''
		for w in self.items:
			w.calc_p()
		source_norm = sum([w.source_p_raw for w in self.items])
		updated_norm = sum([w.updated_p_raw for w in self.items])
		for w in self.items:
			w.normalize(updated_p_norm = updated_norm,source_p_norm = source_norm)

	def _update_prob(self,pdf, verbose = False):
		'''update word probabilities with the pronprobs.'''
		self._reset_update()
		# self.updater_median = np.median([item.logprob for item in pdf.items])
		if len(pdf.items) < self.n_pronprob: raise ValueError(pdf.__str__(),len(pdf.items),'less than expected',self.pronprob_filename,self.__str__())
		self.shift = pdf.items[self.n_pronprob].logprob
		# pdf._shift_logprobs(self.updater_median)
		pdf._shift_logprobs(self.shift)
		hopefulls = [pronprob for pronprob in pdf.items if pronprob.logprob > 0]
		# print('setting',len(hopefulls),'pronprobs')
		bar = pb.ProgressBar()
		bar(range(len(hopefulls)))
		for i,pronprob in enumerate(hopefulls):
			if verbose:
				bar.update(i)
			for word in self.onset2word[pronprob.phon[0]]:
				word.update(pronprob)
		self._calc_p()
		
		
	def calc_cross_entropy(self,pdf):
		'''compute the cross entropy between the word pdf and updated word pdf.
		self should be a word pdf and pdf should be a pronprob pdf
		the word pdf is update with the pronprob pdf
		'''
		if self.pdf_type != 'words_cow' or pdf.pdf_type != 'pronprob':
			raise ValueError(self.pdf_type,pdf.pdf_type,'not defined for update_prob')
		self.pronprob_filename = pdf.filename
		self._update_prob(pdf)
		self.gate = pdf.gate
		self.cross_entropy=stats.entropy([w.prob for w in self.items],[w.updated_p for w in self.items])
		if self.items[0].updated_p != 'na' and type(self.items[0].updated_p) == np.float64: 
			self.ud_entropy = stats.entropy([item.updated_p for item in self.items])

	def plot(self,word= '',prob_type = 'p'):
		''' plot the distribution '''
		self.items.sort(reverse = True)
		if prob_type == 'p': p = np.array([item.prob for item in self.items])
		else: p = np.array([item.logprob for item in self.items])
		if word != '':
			item,index = self.find_item(word)
			plt.vlines(index,np.min(p),np.max(p))
			plt.annotate(item.word + ' ' + str(p[index]) +'  '+ str(index),xy=(index,np.median(p)))
		plt.plot(p)
		plt.grid()
		plt.title(' '.join(self.preceding_words) + '   ' + str(round(self.entropy,2)))
			
		

class lex_word():
	'''Object to match between different pdf's.
	cow pdf / kaldi pdf, words (based on SLM) / phonemes (based on kaldi) 

	word pdf (cow) is a list of words with a probability for each word based on the pre-context
	computed with srilm
	the probability are stored in numpy arrays, the index (word/prob) corresponds to the cow_lex list

	phoneme pdf (kaldi) is a list of phoneme strings (1-8) in length (based on lexicon)
	nbest list (n = 500) of the phoneme string is stored per gate (110 - 650 ms)
	the kaldi pdf contains all possible phoneme strings
	'''
	def __init__(self,line, lexicon, prob = 'NA',kaldi_dict=None):
		self.line = line
		self.lexicon = lexicon
		self.set_prob(prob)
		self.ncol = len(line)
		self.word = line[0]
		if lexicon == 'exp': 
			self.id = int(line[1])
			self.phon = line[2].split(' ')
			self.duration = int(line[3])
			self.index = int(line[-2])
		elif lexicon == 'words_cow':
			self.phon = line[1].split(' ')
			self.index = int(line[-2])
		elif lexicon == 'pronprob':
			self.phon = kaldi_dict[self.word].split(' ')
			self.set_prob(float(line[1]))
			
		self.reset_update()


	def __repr__(self):
		return 'lex_word\t' + self.word + '\t' + ' '.join(self.phon)+'\tp: '+str(self.prob) + '\t' + self.lexicon


	def __lt__(self,other):
		if self.prob == 'NA' or other.prob == 'NA': raise ValueError('probs should be set to sort')
		return self.prob < other.prob

	def set_prob(self,prob=None,logprob=None):
		'''Set probability of the lex word.'''
		if prob != 'NA' and prob != None: 
			self.prob = prob
			self.logprob = np.log10(prob)
		elif logprob != None:
			self.logprob = logprob
			self.prob = 10**logprob
		else:
			self.prob = 'NA'
			self.logprob = 'NA'


	def match(self,other):
		'''Match a word with phoneme string.
		currently all phonemes in the phoneme string should match the phonemes in the word
		graded match could be added (levenstein distance?)
		'''
		assert self.lexicon != other.lexicon
		# pron_var = self.phon if self.lexicon == 'kaldi' else other.phon
		# phon_word = self.phon if self.lexicon == 'cow' else other.phon
		# if len(pron_var) > len(phon_word): return False
		if len(other.phon) > len(self.phon): return False

		match = True
		for i,phon in enumerate(other.phon):
			if phon != self.phon[i]: return False
		# for i,phon in enumerate(pron_var):
			# if phon != phon_word[i]: match = False
		return match


	def update(self,other):
		'''add probability of matched cow or kaldi words to source_update_probs.
		this list can be used to recompute probability for this lex word.
		'''
		if self.match(other):
			self.source_update_probs.append(other)

	def match_onset(self,onset):
		'''Checks whether (first) phoneme matches.'''
		return onset == self.phon[0]

	def reset_update(self):
		'''reset probabilities and source_update_probs (probs of matching lex words.'''
		self.source_update_probs = []
		self.updated_lp = 'NA'
		self.source_lp = 'NA'
		self.updated_p = 'NA'
		self.source_p = 'NA'

	def calc_p(self):
		'''compute new probability based on matches.
		'''
		self.updated_lp_raw = sum([w.logprob for w in self.source_update_probs]) + self.logprob
		self.source_lp_raw = sum([w.logprob for w in self.source_update_probs]) 
		self.updated_p_raw = 10 ** self.updated_lp_raw 
		self.source_p_raw = 10 ** self.source_lp_raw

	def normalize(self,updated_p_norm,source_p_norm):
		'''Create a probability distribution (sums to 1) with specified norms.'''
		self.updated_p = self.updated_p_raw / updated_p_norm
		self.updated_lp = np.log10(self.updated_p)
		self.source_p = self.source_p_raw / source_p_norm
		self.diff_p = self.updated_p - self.prob
		

			
def remove_na(pdf_list):
	'''Remove words that are OOV for the kaldi lexicon.'''
	output = []
	for word in pdf_list:
		if word.phon == ['NA']: continue
		output.append(word)
	return output


def get_pdf_pronprob_filenames(pmn_word, alpha = 0.15):
	'''Get filenames for the word pdf np array and the pronprob nbest file.
	alpha sets a parameter to transform the Kaldi output in to an pdf
	the kaldi output is truncated to an nbest list (n = 500), typically only the top 10 contain somewhat relevant strings.
	'''
	w = pmn_word
	# if w.word.word_number == 0: return False,'first_word_in_sentence',w
	i = int(w.word.pos.token_number) -1
	if i == 0: pdf_filename = path.pdf + 'empty_precontext.pdf.npy'
	else:
		if i > 3: i = 3
		i = i * -1
		pdf_filename = path.pdf + '_'.join(w.precontext[i:]).lower() + '.pdf.npy'
	if not os.path.isfile(pdf_filename): return False,'pdf_file_not_found',pdf_filename
	pronprob_filenames = []
	gates = glob.glob(path.pronprob + '*0')
	for g in gates:
		gate = g.split('/')[-1]
		f = path.pronprob+gate+'/word_'+w.index.lstrip('0') + '_'+gate+'_' + str(alpha).replace('.','') +'.pronprob'
		pronprob_filenames.append( f )
	for f in pronprob_filenames:
		if not os.path.isfile: 
			pronprob_filenames = []
			break
	if len(pronprob_filenames) == 0: return False,'pronprob_files_not_found',f
	return True, pdf_filename, pronprob_filenames


def load_pdf_pronprob(pmn_word, alpha = 0.15):
	'''load the word pdf np array and the pronprob nbest file.
	alpha sets a parameter to transform the Kaldi output in to an pdf
	the kaldi output is truncated to an nbest list (n = 500), typically only the top 10 contain somewhat relevant strings.
	'''
	found,fpdf,fpp = get_pdf_pronprob_filenames(pmn_word, alpha)
	if not found: 
		print(fpdf,fpdf)
		return False
	pdf = 10 ** np.load(fpdf)
	pronprobs = []
	for f in fpp:
		pronprobs.append([[l.split('\t')[0],float(l.split('\t')[1])] for l in open(f).read().split('\n')])
	return pdf,pronprobs


def make_onset_dict(all_phon,cow_lex):
	'''Create a dictionary to speed up kaldi search (only search in set with matching onset).'''
	self.onset2word= {}
	for phon in self.all_phons:
		if phon not in self.onset2kaldi.keys(): self.onset2kaldi[phon] =[]
		for word in cow_lex:
			if word.match_onset(phon): self.onset2kaldi[phon].append(word)


def pmn2cow(pmn_word,kaldi_dict=None,gate='190', n_pronprob = 50):
	'''Create a word pdf based on a pmn_word.'''
	f,word_pdf_fn,pp_fns = get_pdf_pronprob_filenames(pmn_word)
	if not f: return False,'no pdf file found',pmn_word.word
	cow = pdf(path.data +'lexicon_pron_no-na.txt','words_cow',n_pronprob = n_pronprob)
	word_np = pdf(word_pdf_fn,'words_np')
	pp_filename= [f for f in pp_fns if gate in f][0]
	if not os.path.isfile(pp_filename): 
		print(pp_fns,pp_filename)
		return False,'no pronprob file found', pmn_word.word
	pronprob = pdf(pp_filename,'pronprob',kaldi_dict)
	cow.set_pdf(word_np)
	cow.calc_cross_entropy(pronprob)
	return cow, pronprob, word_np
		

def make_kaldi_dict(kaldi_lexicon_name = ''):
	'''create a dictionary that maps a phoneme string name to space seperated phonemes.'''
	if kaldi_lexicon_name == '': kaldi_lexicon_name = path.data + 'lex8.txt'
	kaldi_lex = open_lexicon(kaldi_lexicon_name)
	return dict([line[:2] for line in kaldi_lex])
	

def open_lexicon(name):
	'''open a lexicon.'''
	return [line.split('\t') + [i] for i, line in enumerate(open(name).read().split('\n')) if line]

def open_word_pdf(name):
	'''open propability distribution over 920k words (with cumprob of .9, based on cow.
	only use ~200k which have a phonemic transcription (based on lex4, lexicon made for kaldi by
	mario ganzeboom)
	'''
	return np.load(name)

def open_pron_var_pdf(name):
	'''not used anywhere, remove ???'''
	return np.load(name)

