#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
import annot_chunk 
import codecs
import glob
import sentence 
import match_words
import utils

verbose = False

class Sid:
	'''Speaker object holds the utterances of a speaker in the audio file

	contains a chunk object (annotation structure) and 
	sentence object (for POS tagging and surprisal calc)
	both sentence and chunk contain words, 
	words hold timing info and POS object, surprisal needs to be added
	'''

	def __init__(self,fid = None,sid = 'spreker1',path = '../IFADV_ANNOTATION/ORT/',awd_path= '../IFADV_ANNOTATION/AWD/WORD_TABLES/',corpus = 'IFADV',pos_path = 'POS_IFADV/FROG_OUTPUT/',register = 'spontaneous_dialogue',set_verbose = False):
		'''Speaker object holds the utterances of a speaker in the audio file.
		fid 	file id, filename of the orthographic transcription
		sid 	speaker id, name (in corpus) of the speaker
		path 	directory of ortographic annotations (migrate to path module?)
		awd_... directory of forced aligned annotations
		corpus 	corpus that is the source of the materials
		pos_... directory of the part speech tag annotations
		regi... the type of speech
		set_... whether messages are printed to the screen
		'''
		global verbose
		verbose = set_verbose
		if fid == None:
			fid = 'DVA13U'
			print('calling sid with default file id: ',fid)
		if verbose: print('creating sid with file id:',fid,' and speaker id:',sid)
		self.fid = fid
		self.sid = sid
		self.path = path
		self.awd_path = awd_path
		self.corpus = corpus
		self.pos_path = pos_path
		self.register = register
		self.awd_fn = glob.glob(awd_path + fid + '*' + 'Table')
		if self.corpus == 'IFADV': self.ort_fn = glob.glob(path + fid + '*' + sid)
		elif self.corpus == 'CGN': self.ort_fn = glob.glob(path + fid + '*' + 'Table')
		else:self.ort_fn = []
		self.find_ort_filename()
		self.find_awd_filename()
		self.read_ort()
		self.read_awd()
		self.add_chunks()
		self.nsentences = 0
		self.pos_text_ok = False
		self.npos_sentences_ok = False
		self.make_sentences()
		self.chunk_overlap_indices = []
		self.word_overlap_indices = []
		self.n_chunk_overlap = 0
		self.n_word_overlap = 0
		self.read_frog_pos_output()
		self.split_pos_text()
		self.add_pos_to_sentences() 
		self.ncontent_words = 0
		self.count_content_words()
		utils.make_attributes_available(self,'s',self.sentences)
		utils.make_attributes_available(self,'c',self.chunks)


	def __str__(self):
		a = ['file id:\t' + self.fid ]
		a.append('corpus:\t\t'+str(self.corpus))
		a.append('register:\t'+str(self.register))
		a.append('speaker id:\t'+self.sid)
		a.append('filename:\t'+self.ort_filename)
		a.append('awd filename:\t'+self.awd_filename)
		a.append('pos filename:\t'+self.pos_path)
		a.append('Number of chunks:\t'+ str(self.nchunks))
		a.append('Number of words:\t'+ str(self.nwords))
		a.append('N sentences:\t\t' + str(self.nsentences))
		a.append('N overlap chunks:\t'+ str(self.n_chunk_overlap))
		a.append('N overlap words:\t'+ str(self.n_word_overlap))
		a.append('pos text ok:\t\t'+ str(self.pos_text_ok))
		a.append('npos sentence ok:\t'+ str(self.npos_sentences_ok))
		a.append('ncontent_words\t\t'+ str(self.ncontent_words))
		return '\n'.join(a)

	def set_sid(self,sid = 'spreker1'):
		'''Set speaker id.'''
		if verbose: print('setting speaker id, sid was:',self.sid)
		self.sid = sid
		print('sid now is:',self.sid)
	

	def find_ort_filename(self):
		'''Find file name of ortographic transcription based on fid (file id) and sid (speaker id).
		sid is needed for IFADV, because speakers needed to be seperated
		ort files have been rewritten to table files with praat (orginally textgrids)
		'''
		for line in self.ort_fn:
			if self.fid in line:
				if (self.corpus == 'IFADV' and self.sid in line) or self.corpus == 'CGN':
					self.ort_filename = line
					return line
		print('ort filename not found for: ',self.fid, self.sid,self.path)
		print('fn found',self.ort_fn)
		print('corpus specified:',self.corpus)
		return None

	def find_awd_filename(self):
		'''Find filename of ASR forced aligned start end time of words and phonemes.'''
		for line in self.awd_fn:
			if self.fid in line: 
				self.awd_filename = line
				return line
		print('awd filename not found for: ',self.fid, self.sid)
		return None

	def read_ort(self):
		'''Read in a speaker specific table file (praat exported textgrid).'''
		self.ort_text = [line.split('\t') for line in codecs.open(self.ort_filename,'r','utf8').read().split('\n') if line]
		return self.ort_text

	def read_awd(self):
		'''Read in ASR info (start end times of words and phonems).'''
		self.awd_text = [line.split('\t') for line in codecs.open(self.awd_filename,'r','utf8').read().split('\n') if line and self.sid in line.split('\t')[1] ]
		for line in self.awd_text:
			line[0],line[-1] = float(line[0]),float(line[-1])
		return self.awd_text

	def remove_header(self,ort_text):
		'''Remove header of table file if it is present.'''
		h = ort_text[0]
		if h[0] =='tmin':
			return ort_text[1:] 
		else:
			return ort_text

	def find_awd_items_in_chunk(self,c):
		'''Find awd items that overlap at least 0.04 seconds with the chunk (only needed for CGN files).
			
		overlap time was chosen based on trial and error, index of last awd item is stored to speed up search
		(this works because the table file is in chronological order)
		'''
		awd_items_in_chunk = []
		last = self.last_awd_index
		for i,line in enumerate(self.awd_text[self.last_awd_index:]):
			ok = False
			if line[0] >= c.st and line[-1] <= c.et: ok = True
			if line[0] < c.st and line[-1] > c.st and line[-1] - c.st > 0.04: ok = True
			if line[0] < c.et and line[-1] > c.et and c.et - line[0] > 0.04: ok = True
			if ok:
				if line[2] not in ['sil','sp','_']:
					awd_items_in_chunk.append(line) 
				last = i
		self.last_awd_index = last
		return awd_items_in_chunk

	def add_chunks(self):
		'''Add chunk to the ort object, a chunk is a basic unit of orthografic
		transcription and represented as an object consisting of words.
		'''
		if verbose: print('adding chunks to sid')
		ort_text = self.remove_header(self.ort_text)
		self.chunks = []
		self.nwords,self.nchunks = 0,0
		self.words = []
		self.last_awd_index= 0 
		for i,line in enumerate(ort_text):
			c = annot_chunk.Chunk(line,i,self.ort_filename,self.fid,self.sid,corpus =self.corpus,register = self.register)
			awd_items_in_chunk = self.find_awd_items_in_chunk(c)
			c.add_awd_items_in_chunk(awd_items_in_chunk)
			c.match_awd2word()
			self.nwords += c.nwords
			self.nchunks += 1
			self.chunks.append(c)
			self.words.extend(c.words)


	def make_sentences(self):
		'''Make a sentence from all consecutive words from 1 speaker between eol (. ! ?).'''
		if verbose: print('creating sentences, words between eols')
		self.sentences = []
		sentence_wl = []
		sentence_index = 0
		for w in self.words:
			sentence_wl.append(w)
			if w.eol:
				self.sentences.append(sentence.Sentence(sentence_wl,sentence_index))
				if self.corpus == 'CGN':
					self.sentences[-1].overlap_unknown = False
				sentence_index += 1
				sentence_wl = []
		self.nsentences = len(self.sentences)


	def print_utf8_sentences(self,filename = 'sentences_utf8'):
		'''Create a file with all sentences of this speaker.'''
		filename = filename + '_' + self.fid + '_' + self.sid + '.txt'
		print('saving all sentences to:',filename)
		output = []
		for s in self.sentences:
			output.append(s.string_utf8_words())
		fout = open(filename,'w')
		fout.write('\n'.join(output))
		fout.close()
		

	def read_frog_pos_output(self):
		'''Read in a file with FROG POS output, which was based on the sentences created
		with print_utf8_sentences.
		'''
		fn = glob.glob(self.pos_path +'*'+self.fid+'_'+self.sid+'*')
		if len(fn) != 1:
			print('did not find 1 filename, maybe no FROG POS output present')
			print('found the following using',self.fid,self.sid,self.pos_path,':')
			print('file(s) found:',fn)
			self.pos_text_ok = False
			return 0
		if verbose: print('reading FROG output pos tags, with',self.fid,self.sid,'\nfilename:',fn[0])
		self.pos_text_ok = True
		self.pos_filename = fn[0]
		self.pos_text = codecs.open(self.pos_filename,'r','utf8').read().split('\n')


	def split_pos_text(self):
		'''Split the Frog output into a list of sentences. 
		Each sentence is a list of words and as word is a list containing pos info.
		'''
		if self.pos_text_ok == False:
			print('pos tags not read in')
			self.npos_sentences_ok = False
			return 0
		if verbose: print('splitting pos text into sentences')
		self.pos_sentences = []
		pos_wl = []
		for line in self.pos_text:
			if line == '' and pos_wl != []:
				self.pos_sentences.append(pos_wl)
				pos_wl = []
			elif line != '':
				pos_wl.append(line.split('\t'))
		if self.nsentences == len(self.pos_sentences): self.npos_sentences_ok = True
		else: self.npos_sentences_ok = False
				
				
	def add_pos_to_sentences(self):
		'''Adding pos object (with pos info) to each word in the sentence.'''
		if verbose: print('adding pos tags to words in sentences')
		if self.pos_text_ok == False:
			print('pos tags not read in')
			return 0
		if self.npos_sentences_ok == False:
			print('npos sentences do not equal number of sentences')
			return 0
		for i,s in enumerate(self.sentences):
			pos_tags = self.pos_sentences[i]
			s.add_pos_to_words(pos_tags)
					
	
	def count_content_words(self):
		'''Counts the number of words that are content words.'''
		self.ncontent_words = 0
		for w in self.words:
			if w.pos_ok and w.pos.content_word:
				self.ncontent_words += 1

