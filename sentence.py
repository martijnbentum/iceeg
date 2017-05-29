import pos

class Sentence:
	def __init__(self,words,sentence_number = 0):
		self.words = words
		self.nwords = len(words)
		self.sentence_number = sentence_number
		self.st = self.words[0].st
		self.et = self.words[-1].et
		self.duration = self.et - self.st
		self.find_chunk_numbers()
		self.check_sentence()
		self.npos_ok = False


	def __str__(self):
		a = ['sentence:\t'+self.string_words()]
		a.append('nwords:\t'+str(self.nwords))
		a.append('start_time:\t'+str(self.st))
		a.append('end_time:\t'+str(self.et))
		a.append('duration:\t'+str(self.duration))
		a.append('nchunks:\t'+str(self.n_chunks))
		a.append('chunk_numbers:\t'+' '.join(map(str,self.chunk_numbers)))
		a.append('sentence ok:\t'+str(self.ok))
		a.append('npos_ok:\t'+str(self.npos_ok))
		return '\n'.join(a)


	def print_words(self):
		print(self)
		print('-'*50)
		for w in self.words:
			print(w)
			print('-'*50)


	def string_words(self):
		output = [] 
		for w in self.words:
			output.append( w.word )
		return ' '.join(output)


	def string_utf8_words(self):
		output = []
		for w in self.words:
			output.append( w.word_utf8_nocode )
		return ' '.join(output)


	def find_chunk_numbers(self):
		self.chunk_numbers = []
		for w in self.words:
			if w.chunk_number not in self.chunk_numbers:
				self.chunk_numbers.append(w.chunk_number)
		self.n_chunks = len(self.chunk_numbers)
		

	def check_sentence(self):
		self.ok = True
		for i,w in enumerate(self.words):
			if w.eol == True and i != (len(self.words) - 1):
				self.ok = False
		

	def add_pos_to_words(self,pos_tags):
		if self.nwords == len(pos_tags):
			self.npos_ok = True
			for wi,pos_tag in enumerate(pos_tags):
				w = self.words[wi]
				w.pos = pos.Pos(pos_tag,self.sentence_number)
				if w.word_utf8_nocode == w.pos.token: w.pos_ok = True
				else: w.pos_ok = False
