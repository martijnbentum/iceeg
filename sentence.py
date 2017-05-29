class sentence:
	def __init__(self,words,sentence_number = 0):
		self.words = words
		self.sentence_number = sentence_number
		self.st = self.words[0].st
		self.et = self.words[-1].et
		self.duration = self.et - self.st
		self.find_chunk_numbers()
		self.check_sentence()


	def __str__(self):
		s = ['sentence:\t'+self.string_words()]
		s.append('start_time:\t'+str(self.st))
		s.append('end_time:\t'+str(self.et))
		s.append('duration:\t'+str(self.duration))
		s.append('nchunks:\t'+str(self.n_chunks))
		s.append('chunk_numbers:\t'+' '.join(map(str,self.chunk_numbers)))
		s.append('sentence ok:\t'+str(self.ok))
		return '\n'.join(s)

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

	def string_output_words(self):
		output = [] 
		for w in self.words:

			if '*' in w.word: ow = w.word.split('*')[0]
			else: ow = w.word
			ow.replace('.','').replace('?','').replace('!','')
				
			output.append( ow )
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
		
