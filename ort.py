#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
import codecs
import glob
import sentence as s
import match_words

class Ort:
	def __init__(self,fid = None,sid = 'spreker1', path = '../IFADV_ANNOTATION/ORT/',awd_path = '../IFADV_ANNOTATION/AWD/WORD_TABLES/'):
		if fid == None:
			fid = 'DVA13U'
			print('calling ort class with default file id: ',fid)
		print('creating ort with file id:',fid,' and speaker id:',sid)
		self.fid = fid
		self.path = path
		self.awd_path = awd_path
		self.sids = []
		self.speakers_present = False
		self.nspeakers = 0
		self.speakers = [] 

		if sid:
			self.add_speaker(sid)

	def __str__(self):
		s = ['file id:\t' + self.fid ]
		s.append('speaker ids:\t'+'  '.join(self.sids))
		for speaker in self.speakers:
			s.append('-'* 50)
			s.append(speaker.__str__())
		return '\n'.join(s)
		

	def add_speaker(self,sid):
		self.speakers.append( Sid(self.fid,sid,self.path) )
		self.sids.append(sid)
		self.speakers_present = True
		self.nspeakers += 1


	def find_awd_filename(self):
		awd_fn = glob.glob(self.awd_path + '*' + '.Table')
		for f in awd_fn:
			if self.fid in f:
				self.awd_filename = f
				return 1
		self.awd_filename = None	
		return 0


	def check_overlap(self):
		if len(self.speakers) != 2:
			print('need 2 speaker to check for overlap (> 2 speakers not implemented')
			return 0
		s1,s2 = self.speakers
		for i,w1 in enumerate(s1.words):
			w1.set_overlap(overlap = False)
			for i2,w2 in enumerate(s2.words):
				w2.set_overlap(overlap = False)
				if w1.st and w1.et and w2.st and w2.et:
					if (w1.st >= w2.st and w1.st < w2.et) \
						or (w1.et >= w2.st and w1.et < w2.et):
						if i not in s1.word_overlap_indices:
							s1.word_overlap_indices.append(i)
						if i2 not in s2.word_overlap_indices:
							s2.word_overlap_indices.append(i2)

		for i in s1.word_overlap_indices:
			s1.words[i].set_overlap(overlap = True)
		for i in s2.word_overlap_indices:
			s2.words[i].set_overlap(overlap = True)

		for s in self.speakers:
			for i,c in enumerate(s.chunks):
				c.check_overlap()
				if c.overlap and not c.overlap_unknown:
					s.chunk_overlap_indices.append(i)

		s1.n_chunk_overlap = len(s1.chunk_overlap_indices)
		s1.n_word_overlap = len(s1.word_overlap_indices)
		s2.n_chunk_overlap = len(s2.chunk_overlap_indices)
		s2.n_word_overlap = len(s1.word_overlap_indices)

	def make_sentences(self):
		for s in self.speakers:
			s.make_sentences()

	def print_sentences(self):
		for s in self.speakers:
			s.print_utf8_sentences()

class Sid:
	def __init__(self,fid = None,sid = 'spreker1',path = '../IFADV_ANNOTATION/ORT/',awd_path= '../IFADV_ANNOTATION/AWD/WORD_TABLES/'):
		if fid == None:
			fid = 'DVA13U'
			print('calling ort class with default file id: ',fid)
		print('creating ort with file id:',fid,' and speaker id:',sid)
		self.fid = fid
		self.sid = sid
		self.path = path
		self.awd_path = awd_path
		self.awd_fn = glob.glob(awd_path + fid + '*' + 'Table')
		self.ort_fn = glob.glob(path + fid + '*' + sid)
		self.find_ort_filename()
		self.find_awd_filename()
		self.read_ort()
		self.read_awd()
		self.add_chunks()
		self.make_sentences()
		self.chunk_overlap_indices = []
		self.word_overlap_indices = []
		self.n_chunk_overlap = 0
		self.n_word_overlap = 0

	def __str__(self):
		s = ['file id:\t' + self.fid ]
		s.append('speaker id:\t'+self.sid)
		s.append('filename:\t'+self.ort_filename)
		s.append('awd filename:\t'+self.awd_filename)
		s.append('Number of chunks:\t'+ str(self.nchunks))
		s.append('Number of words:\t'+ str(self.nwords))
		s.append('N sentences:\t\t' + str(self.nsentences))
		s.append('N overlap chunks:\t'+ str(self.n_chunk_overlap))
		s.append('N overlap words:\t'+ str(self.n_word_overlap))
		return '\n'.join(s)
	

	def find_ort_filename(self):
		for line in self.ort_fn:
			if self.fid in line and self.sid in line:
				self.ort_filename = line
				return line
		print('ort filename not found for: ',self.fid, self.sid)
		return None

	def find_awd_filename(self):
		for line in self.awd_fn:
			if self.fid in line: 
				self.awd_filename = line
				return line
		print('awd filename not found for: ',self.fid, self.sid)
		return None

	def read_ort(self):
		#read in a speaker specific table file (praat exported textgrid)
		self.ort_text = [line.split('\t') for line in codecs.open(self.ort_filename,'r','utf8').read().split('\n') if line]
		return self.ort_text

	def read_awd(self):
		self.awd_text = [line.split('\t') for line in codecs.open(self.awd_filename,'r','utf8').read().split('\n') if line and self.sid in line.split('\t')[1] ]
		return self.awd_text

	def remove_header(self,ort_text):
		h = ort_text[0]
		if h[0] =='tmin':
			return ort_text[1:] 
		else:
			return ort_text

	def find_awd_items_in_chunk(self,c):
		awd_items_in_chunk = []
		for line in self.awd_text:
			ok = False
			if float(line[0]) >= c.st and float(line[-1]) <= c.et: ok = True
			if float(line[0]) < c.st and float(line[-1]) > c.st: ok = True
			if float(line[0]) < c.et and float(line[-1]) > c.et: ok = True
			if ok:
				if line[2] not in ['sil','sp']:
					awd_items_in_chunk.append(line) 
		return awd_items_in_chunk

	def add_chunks(self):
		# add chunk to the ort object, a chunk is a basic unit of orthografic
		# transcription and represented as an object consisting of words
		ort_text = self.remove_header(self.ort_text)
		self.chunks = []
		self.nwords,self.nchunks = 0,0
		self.words = []
		for i,line in enumerate(ort_text):
			c = Chunk(line,i,self.ort_filename,self.fid,self.sid)
			awd_items_in_chunk = self.find_awd_items_in_chunk(c)
			c.add_awd_items_in_chunk(awd_items_in_chunk)
			c.match_awd2word()
			self.nwords += c.nwords
			self.nchunks += 1
			self.chunks.append(c)
			self.words.extend(c.words)

	def make_sentences(self):
		self.sentences = []
		sentence = []
		sentence_index = 0
		for w in self.words:
			sentence.append(w)
			if w.eol:
				self.sentences.append(s.sentence(sentence,sentence_index))
				sentence_index += 1
				sentence = []
		self.nsentences = len(self.sentences)
				
	def print_sentences(self,filename = 'sentences'):
		filename = filename + self.fid + '_' + self.sid + '.txt'
		print('saving all sentences to:',filename)
		output = []
		for s in self.sentences:
			output.append(s.string_output_words())
		fout = open(filename,'w')
		fout.write('\n'.join(output))
		fout.close()

	def print_utf8_sentences(self,filename = 'sentences_utf8'):
		filename = filename + '_' + self.fid + '_' + self.sid + '.txt'
		print('saving all sentences to:',filename)
		output = []
		for s in self.sentences:
			output.append(s.string_utf8_words())
		fout = open(filename,'w')
		fout.write('.\n'.join(output))
		fout.close()
		


class Chunk:
	def __init__(self,line,chunk_number,filename = None, fid = None,sid = None,cid = None):
		self.line = line
		self.chunk_number = chunk_number
		self.add_info(filename,fid,sid,cid)
		self.check_line()
		self.st, self.tier, self.label, self.et = self.line
		self.fix_label()
		self.duration = self.et - self.st
		self.chunk_id = 'default_id'
		self.add_words()
		self.check_overlap()
		self.add_awd_items_in_chunk()
		self.nwords = len(self.words)


	def __str__(self):
		s = ['chunk:\t\t'+self.label]
		s.append( 'nwords:\t\t'+str(self.nwords))
		s.append( 'awd chunk:\t'+self.string_awd())
		s.append( 'n_awd_words:\t'+str(self.n_awd_words))
		s.append( 'start time:\t'+ str(self.st))
		s.append( 'end time:\t'+ str(self.et))
		s.append( 'duration:\t'+ str(self.duration))
		s.append( 'index:\t\t'+str(self.chunk_number ))
		s.append( 'file id:\t'+self.fid) 
		s.append( 'speaker id:\t'+self.sid) 
		s.append( 'overlap:\t'+str(self.overlap)) 
		s.append( 'overlap unk:\t'+str(self.overlap_unknown)) 
		return '\n'.join(s)

	def check_line(self):
		if type(self.line) == str:
			temp = line.split('\t')
			if len(temp) == 4:
				self.line = temp 
		if type(self.line) != list and self.line != 4:
			dl =[0.0,'default-speaker','default label',0.0]
			self.line = dl
		try:
			st,et = float(self.line[0]),float(self.line[-1])
			self.line[0],self.line[-1] = st,et
		except:
			self.line = dl


	def fix_label(self):
		# checks for interpuction and special code problems and fixes those
		# a problem is a space between .?* and word or the presence of a , 

		self.label_problem = []
		for problem in [' .',',',' ?',' *','  ']:
			if problem in self.label: self.label_problem.append(problem)  
				
		if self.label_problem != []:
			self.label = self.label.replace(' .','.').replace(',','').replace(' *','*').replace(' ?','?').replace('  ',' ')
		
		if self.label[-1] == ' ':
			self.label = self.label[:-1]
		if self.label[0] == ' ':
			self.label = self.label[1:]


	def add_info(self,filename = None, fid = None,sid = None,cid = None):
		if filename: self.filename = filename
		else: self.filename = None
		if fid: self.fid = fid
		else: self.fid = None
		if sid: self.sid = sid
		else: self.sid = None
		if cid: self.cid = cid
		else: self.cid = None

	def add_words(self):
		words = self.label.split(' ')
		self.words = []
		for i,w in enumerate(words):
			if w:
				w =Word(w,i,self.chunk_number,self.st,self.et,self.filename,self.fid,self.sid,self.cid)
				self.words.append(w)

	def check_overlap(self):
		self.overlap_unknown,self.overlap = False,False
		self.overlap_indices = []
		for i,w in enumerate(self.words):
			if w.overlap_unknown: self.overlap_unknown = True
			if w.overlap == True: 
				self.overlap = True
				self.overlap_indices.append(i)

	def add_awd_items_in_chunk(self,awd_items = None):
		if awd_items == None:
			self.awd_words = None
			self.awd_phon_words = None
			self.awd_phon= None
			self.n_awd_words,self.n_awd_phon_words,self.n_awd_phon = 0, 0, 0
			return 0
		self.awd_words = [line for line in awd_items if 'ort-word' in line[1]]
		self.awd_phon_words = [line for line in awd_items if 'phon-word' in line[1]]
		self.awd_phon= [line for line in awd_items if 'phon-phon' in line[1]]
		w,pw,p = len(self.awd_words), len(self.awd_phon_words), len(self.awd_phon)
		self.n_awd_words,self.n_awd_phon_words,self.n_awd_phon = w, pw, p

	def string_awd(self,awd_type = 'words'):
		if awd_type == 'words':
			if self.awd_words:
				return ' '.join([line[2] for line in self.awd_words])
		if awd_type == 'phon_words':
			if self.awd_words:
				return ' '.join([line[2] for line in self.awd_phon_words])
		if awd_type == 'phon':
			if self.awd_phon:
				return ' '.join([line[2] for line in self.awd_phon])
		

	def match_awd2word(self):
		awd_word_list = [line[2] for line in self.awd_words]
		self.matcher = match_words.Matcher(self.label,awd_word_list)
		for i,w in enumerate(self.words):
			if i in self.matcher.ort_index2awd_index.keys():
				awd_index = self.matcher.ort_index2awd_index[i]
				st = float(self.awd_words[awd_index][0])
				et = float(self.awd_words[awd_index][-1])
				w.add_times(st = st, et = et)
				w.awd_ok = True
			else:
				w.add_times(st = self.st, et = self.et)
				w.awd_ok = False
		if self.matcher.ratio < 1:
			print('not a complete match between ort and awd')
			print(self.matcher)

		

class Word:
	def __init__(self, word,word_number, chunk_number, chunk_st,chunk_et,filename = None, fid = None,sid = None,cid = None,wid = None, st = None, et = None):
		self.word = word
		self.word_number = word_number
		self.chunk_number= chunk_number
		self.chunk_st = chunk_st
		self.chunk_et = chunk_et
		self.status()
		self.filename = filename
		self.fid = fid
		self.sid = sid
		self.cid = cid
		self.st = st
		self.et = et
		if st and et: self.duration = et - st
		else: self.duration = None
		self.add_info(filename,fid,sid,cid,wid)
		self.add_times()
		self.prev_word = None
		self.next_word = None 
		self.set_overlap()
		self.replace_diacritics()
		self.remove_special_code()

	def __str__(self):
		s = ['word:\t\t'+self.word]
		s.append( 'word index:\t'+str(self.word_number) )
		s.append( 'start time:\t'+str(self.st) )
		s.append( 'end time:\t'+str(self.et) )
		s.append( 'duration:\t'+str(self.duration) )
		s.append( 'chunk index:\t'+str(self.chunk_number) )
		s.append( 'chunk st:\t'+str(self.chunk_st) )
		s.append( 'chunk et:\t'+str(self.chunk_et) )
		s.append( 'special:\t'+str(self.special_code))
		s.append( 'punctuation:\t'+str(self.punctuation))
		s.append( 'end of line:\t'+str(self.eol))
		s.append( 'file id:\t'+self.fid) 
		s.append( 'speaker id:\t'+self.sid) 
		s.append( 'overlap:\t'+str(self.overlap)) 
		s.append( 'overlap unk:\t'+str(self.overlap_unknown)) 
		return '\n'.join(s)
		

	def status(self):
		import string
		self.special_code,self.punctuation,self.eol = False,False,False
		if '*' in self.word:
			self.special_code = True
		for p in string.punctuation.replace('*',''):
			if p in self.word:
				self.punctuation = True
		for eol in ['.','!','?']:
			if eol in self.word:
				self.eol = True

	def add_info(self,filename = None, fid = None,sid = None,cid = None,wid = None,st = None, et = None):
		if filename: self.filename = filename
		if fid: self.fid = fid
		if sid: self.sid = sid
		if cid: self.cid = cid
		if wid: self.wid = wid
		if st: self.st = st
		if et: self.et = et

	def add_prev_word(self,word = None):
		self.prev_word = word

	def add_next_word(self,word = None):
		self.next_word = word

	def set_overlap(self,overlap = None):
		if overlap == None:
			self.overlap_unknown = True
			self.overlap = False
		else:
			self.overlap = overlap 
			self.overlap_unknown = False

	def add_times(self,st = None, et= None):
		if st: self.st = st
		if et: self.et = et
		if self.st and self.et:
			self.duration = self.et - self.st

	
	def replace_diacritics(self):
		rd = {'\e^':'ê','\e"':'ë',"\e'":'é',"\e`":'è','\i"':'ï',"\i'":'í','\\u"':'ü',"\\u'":'ú',"\\a'":'á',"\\a`":'à','\o"':'ö',"\o'":'ó','\c,':'ç'}
		temp = self.word
		for c in rd.keys():
			temp = temp.replace(c,rd[c])
		self.word_utf8 = temp


	def remove_special_code(self):
		if self.special_code:
			self.word_nocode = self.word.split('*')[0]
			self.word_utf8_nocode = self.word_utf8.split('*')[0]
		else:
			self.word_nocode = self.word
			self.word_utf8_nocode = self.word_utf8

		for eol in ['.','!','?']:
			self.word_nocode = self.word_nocode.replace(eol,'')
			self.word_utf8_nocode = self.word_utf8_nocode.replace(eol,'')
