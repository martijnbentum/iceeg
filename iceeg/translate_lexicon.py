import glob
import kaldi2probs
import numpy as np
import os
import path
import pmn_input_data as pid
import progressbar as pb
from scipy import stats


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


class lexicon_translate():
	'''Provides links between different lexicons (cow kaldi) and pdf of words and phoneme strings.'''
	def __init__(self,cow_lex_fn = None,kaldi_lex_fn = None,exp_lex_fn = None):
		'''Create lexicon transelate object based on the lexicon filename.
		cow_lex 	the word list to for the pdfs for different pre-contexts
					pdf is:
					np array of probabilities for words in cow_lex, given a specific pre-context
					index in np array matches index of cow_lex word
		kaldi_lex 	the possible phoneme strings kaldi used to transcribe the gates
		exp_lex 	the word list of the EEG-experiment
		'''
		if exp_lex_fn == None: self.exp_lex_fn = path.data + 'experiment_lexicon_w2i-pron-dur.txt'
		self.exp_lex = open_lexicon(self.exp_lex_fn)
		if cow_lex_fn == None: self.cow_lex_fn = path.data + 'lexicon_pron.txt'
		self.cow_lex = open_lexicon(self.cow_lex_fn)
		if kaldi_lex_fn == None: self.kaldi_lex_fn = path.data + 'lex8.txt'
		self.kaldi_lex = open_lexicon(self.kaldi_lex_fn)
		#pmn contains all words of the experiment pmn_words and can create the filename for pronprob
		self.pmn = pid.pmn_data(history_length =3)
		# create lex_words for exp cow and kaldi for matching purposes
		self.make_pdf()
		# remove all cow words that do not have phonemic transcription
		self.cow_pdf = remove_na(self.cow_pdf)
		# show phoneme set of kaldi dict
		self.all_phons = list(set(' '.join([' '.join(w.phon) for w in self.cow_pdf]).split(' ')))
		# make a dictionary to speed up matching cow ~ kaldi
		self.make_onset_dict()


	def make_onset_dict(self):
		'''Create a dictionary to speed up kaldi search (only search in set with matching onset).'''
		self.onset2kaldi = {}
		for phon in self.all_phons:
			if phon not in self.onset2kaldi.keys(): self.onset2kaldi[phon] =[]
			for word in self.kaldi_pdf:
				if word.match_onset(phon): self.onset2kaldi[phon].append(word)
					

	def make_pdf(self):
		'''Create lex word for each lexicon, which can be used for matching.
		matching is used to updat probability between words (cow) and phoneme (kaldi) probs
		'''
		self.cow_pdf, self.kaldi_pdf,self.exp_pdf = [],[],[]
		for line in self.exp_lex:
			self.exp_pdf.append(lex_word(line,'exp'))
		for line in self.cow_lex:
			self.cow_pdf.append(lex_word(line,'cow'))
		for line in self.kaldi_lex:
			self.kaldi_pdf.append(lex_word(line,'kaldi'))


	def connect_cow_kaldi(self):
		'''Match each cow word with all matching kaldi words (phoneme strings).
		match is each phoneme in kaldi word should be in cow word left to right
		matches are used to update probs between words (cow) and phoneme (kaldi) probs
		and subsequently compute entropy
		'''
		bar = pb.ProgressBar()
		bar(range(len(self.cow_pdf)))
		# connect all words to phoneme strings in kaldi
		for i,coww in enumerate(self.cow_pdf):
			bar.update(i)
			for kaldiw in self.onset2kaldi[coww.phon[0]]:
				coww.update(kaldiw)
		# connect all phoneme strings in kaldi with cow words
		for w in self.cow_pdf:
			for kw in w.source_update_probs:
				kw.source_update_probs.append(w)

		

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
	def __init__(self,line, lexicon, prob = 'NA'):
		self.line = line
		self.lexicon = lexicon
		self.set_prob(prob)
		self.ncol = len(line)
		self.word = line[0]
		if lexicon == 'exp': 
			self.id = int(line[1])
			self.phon = line[2].split(' ')
			self.duration = int(line[3])
		else: self.phon = line[1].split(' ')
		self.index = line[-1]
		self.reset_update()


	def __repr__(self):
		return 'lex_word\t' + self.word + '\t' + ' '.join(self.phon)+'\tp: '+str(self.prob) + '\t' + self.lexicon

	def set_prob(self,prob):
		'''Set probability of the lex word.'''
		if prob != 'NA': 
			self.prob = prob
			self.logprob = np.log10(prob)
		else:
			self.prob = 'NA'
			self.logprob = 'NA'


	def match(self,other):
		'''Match a word with phoneme string.
		currently all phonemes in the phoneme string should match the phonemes in the word
		graded match could be added (levenstein distance?)
		'''
		assert self.lexicon != other.lexicon
		pron_var = self.phon if self.lexicon == 'kaldi' else other.phon
		phon_word = self.phon if self.lexicon == 'cow' else other.phon
		if len(pron_var) > len(phon_word): return False

		match = True
		for i,phon in enumerate(pron_var):
			if phon != phon_word[i]: match = False
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
		self.updated_p = self.updated_p_raw / updated_p_norm
		self.source_p = self.source_p_raw / source_p_norm
		

			
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
	if w.word.word_number == 0: return False,'first_word_in_sentence'
	i = w.word.word_number * -1
	pdf_filename = path.pdf + '_'.join(w.precontext[i:]).lower() + '.pdf.npy'
	if not os.path.isfile(pdf_filename): return False,'pdf_file_not_found',pdf_filename
	f = path.pronprob+'word_'+w.index.lstrip('0') + '_*_' + str(alpha).replace('.','') +'.pronprob'
	pronprob_filenames = glob.glob(f)
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


def make_kaldi_index_dict(kaldi_pdf):
	'''returns a dictionary that holds the index of each kaldi 'word' (phoneme string without spaces).'''
	d = {}
	for index,w in enumerate(kaldi_pdf):
		d[w.word] = index
	return d
	
def set_kaldi_probs(kaldi_pdf,pronprob):
	'''Set the probs in pronprobs on the kaldi 'words', these are matched with the words and can thus update words.'''
	d = make_kaldi_index_dict(kaldi_pdf)
	smallest_p = min([line[1] for line in pronprob])
	for word,p in pronprob:
		kaldi_pdf[d[word]].set_prob(p)
	for w in kaldi_pdf:
		if w.prob == 'NA': w.set_prob(smallest_p)
	# for index in indices:
		# kaldi_pdf[index].set_prob(smallest_p)

def set_cow_probs(cow_pdf,pdf):
	'''Set the word probs based on the word pdf given the preceding context (preceding words upto 3).
	'''
	for w in cow_pdf:
		w.set_prob(pdf[w.index])
	
def calc_p(pdf):
	'''Calc the new pdf given the update.'''
	for w in pdf:
		w.calc_p()
	source_norm = sum([w.source_p_raw for w in pdf])
	updated_norm = sum([w.updated_p_raw for w in pdf])
	for w in pdf:
		w.normalize(updated_p_norm = updated_norm,source_p_norm = source_norm)
	
def calc_entropy(t):
	'''calculate the entropy of the pdf pre and post update for words and kaldiwords (phoneme strings).'''
	cow_pdf, kaldi_pdf = t.cow_pdf, t.kaldi_pdf
	t.cow_p = [w.prob for w in cow_pdf]
	t.cow_source_p = [w.source_p for w in cow_pdf]
	t.cow_updated_p = [w.updated_p for w in cow_pdf]

	t.cow_entropy = stats.entropy(t.cow_p)
	t.cow_source_entropy = stats.entropy(t.cow_p,t.cow_source_p)
	t.cow_updated_entropy = stats.entropy(t.cow_p,t.cow_updated_p)

	t.kaldi_p = [w.prob for w in kaldi_pdf]
	t.kaldi_source_p = [w.source_p for w in kaldi_pdf]
	# t.kaldi_updated_p = [w.updated_p for w in kaldi_pdf]

	t.kaldi_entropy = stats.entropy(t.kaldi_p)
	t.kaldi_source_entropy = stats.entropy(t.kaldi_source_p,t.kaldi_p)
	# t.kaldi_updated_entropy = stats.entropy(t.kaldi_updated_p, t.kaldi_p)



		
	
