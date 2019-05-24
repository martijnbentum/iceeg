#helper module to extract cross entropy values for word in the experiment
import os
import path
import utils

header = open(path.pmn_datasets + 'header').read().strip('\n').split('\t')
dwc = utils.load_dict_word_code2pmn_index()

def word2info(w,gate = 190):
	'''use word object to load the entropy updated entropy cross entropy 
	and updated surprisal information, also generates the word_code and word_index.
	'''
	word = w.word_utf8_nocode_nodia()
	word_code, word_index,pmn_word = 'NA','NA',None
	if hasattr(w,'pos'):
		word_code = utils.make_word_code(w)
		word_index = dwc[word_code]
		pmn_word = get_pmn_word(word_index +'_'+word,gate)
	if pmn_word == None: output = ['NA'] * 5
	else: output = get_all(pmn_word)
	return word_code,word_index,output

def get_pmn_words(f):
	pmn_words = []
	bads = []
	for g in range(110,660,20):
		pmn_word = get_pmn_word
		if pmn_word == None and f not in bads: bads.append(f)
		else: pmn_words.append(pmn_word)
	if len(bads) > 0: print('some file were not found:',bads)
	return pmn_words, bads

def get_pmn_word(f,gate):
	filename = path.pmn_datasets + str(gate) + '/WORDS/' + f
	pmn_word = filename2pmn_word(filename)
	if filename == pmn_word: return None
	return pmn_word 

def filename2pmn_word(filename):
	if not os.path.isfile(filename): return filename
	else: return open(filename).read().strip('\n').split('\t')

def get(pmn_word,column_name):
	pmn_word = check_pmn_word(pmn_word)
	index = header.index(column_name)
	return pmn_word[index]
	
def get_entropy(pmn_word):
	return get(pmn_word,'entropy')

def get_updated_entropy(pmn_word):
	return get(pmn_word,'ud_entropy')

def get_cross_entropy(pmn_word):
	return get(pmn_word,'cross_entropy')

def get_logprob(pmn_word):
	return get(pmn_word,'surprisal')

def get_updated_logprob(pmn_word):
	return get(pmn_word,'ud_surprisal')

def get_all(pmn_word):
	'''returns the entropy, updated entropy, cross entropy and updated surprisal. 
	pmn_word is found in pmn_datasets and describes the info of a specific word in the experiment
	this function extracts the relevant information from this list
	'''
	w = check_pmn_word(pmn_word)
	o = get_entropy(w),get_updated_entropy(w),get_cross_entropy(w),get_logprob(w),get_updated_logprob(w)
	return o

def check_pmn_word(pmn_word):
	if type(pmn_word) == str:
		return pmn_word.split('\t')
	else: return pmn_word
