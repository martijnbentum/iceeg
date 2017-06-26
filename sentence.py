import pos

class Sentence:
	'''A structure containing all words by one speaker between eol charachters ... . ! ?

	a sentence contains word objects which also link to the chunk object
	(chunk object reflect the ortogrpahic annotation structure)
	'''
	
	def __init__(self,words,sentence_number = 0):
		'''Make sentence object from a list of word object that form a sentence'''
		self.words = words
		self.nwords = len(words)
		self.sentence_number = sentence_number
		self.st = self.words[0].st
		self.et = self.words[-1].et
		try: self.duration = round(self.et - self.st,3)
		except: self.duration = -0.999
		self.find_chunk_numbers()
		self.check_sentence()
		self.npos_ok = False


	def __str__(self):
		a = ['sentence:\t'+self.string_words()]
		a.append('nwords:\t\t'+str(self.nwords))
		a.append('start_time:\t'+str(self.st))
		a.append('end_time:\t'+str(self.et))
		a.append('duration:\t'+str(self.duration))
		a.append('nchunks:\t'+str(self.n_chunks))
		a.append('chunk_numbers:\t'+' '.join(map(str,self.chunk_numbers)))
		a.append('sentence ok:\t'+str(self.ok))
		a.append('npos_ok:\t'+str(self.npos_ok))
		return '\n'.join(a)


	def print_words(self):
		'''Print the info of each word'''
		print(self)
		print('-'*50)
		for w in self.words:
			print(w)
			print('-'*50)


	def string_words(self):
		'''Create an ascii sentence out of all words in the sentence'''
		output = [] 
		for w in self.words:
			output.append( w.word )
		return ' '.join(output)


	def string_utf8_words(self):
		'''Create an utf8 sentence out of all words in the sentence'''
		output = []
		for w in self.words:
			output.append( w.word_utf8_nocode )
		return ' '.join(output)


	def find_chunk_numbers(self):
		'''Create a list of all chunk numbers of the words in the sentence
		(this does not mean all words of these chunks are in this sentence)
		'''
		self.chunk_numbers = []
		for w in self.words:
			if w.chunk_number not in self.chunk_numbers:
				self.chunk_numbers.append(w.chunk_number)
		self.n_chunks = len(self.chunk_numbers)
		

	def check_sentence(self):
		'''Check whether last word is an eol word (this should be the case)'''
		self.ok = True
		for i,w in enumerate(self.words):
			if w.eol == True and i != (len(self.words) - 1):
				self.ok = False
		

	def add_pos_to_words(self,pos_tags):
		'''Add POS tag info to each word

		Uses the FROG POS output and assumes that the output has the identical 
		sentence structure as the object (number of words and order)
		'''
		if self.nwords == len(pos_tags):
			self.npos_ok = True
			for wi,pos_tag in enumerate(pos_tags):
				w = self.words[wi]
				w.pos = pos.Pos(pos_tag,self.sentence_number)
				if w.word_utf8_nocode == w.pos.token: w.pos_ok = True
				else: w.pos_ok = False
