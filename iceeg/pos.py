
class Pos:
	'''Create a pos object from a Frog output line (corresponding with 1 word).''' 

	def __init__(self,pos_line = None,sentence_number = -999):
		'''Create a pos object from a Frog output line (corresponding with 1 word).
		Frog output line is created with Frog: https://languagemachines.github.io/frog/
		''' 
		if pos_line == None:
			print('expected a line from frog output like so (can also be a list):')
			print('1   want    want    [want]  VG(neven)   0.998782    O   B-CONJP')
			return 0
		self.pos_line = pos_line
		self.set_info()
		self.sentence_number = sentence_number # if no number provide default to -999


	def __str__(self):
		a=['token:\t\t\t'+self.token]
		a.append('pos:\t\t\t'+self.pos)
		a.append('simple_pos:\t\t'+self.pos_simple)
		a.append('content word:\t\t'+str(self.content_word)) 
		a.append('lemma:\t\t\t'+self.lemma)
		a.append('prob tag:\t\t'+ self.probability_of_tag)
		a.append('token number:\t\t'+self.token_number)
		return '\n'.join(a)


	def print_all_info(self):
		a=['token:\t\t\t'+self.token]
		a.append('pos:\t\t\t'+self.pos)
		a.append('simple_pos:\t\t\t'+self.pos_simple)
		a.append('lemma:\t\t\t'+self.lemma)
		a.append('morph seg:\t\t'+self.morphological_segmentation)
		a.append('prob tag:\t\t'+ self.probability_of_tag)
		a.append('token number:\t\t'+self.token_number)
		a.append('named_entity:\t\t'+self.named_entity)
		a.append('base_phrase_chunk:\t'+self.base_phrase_chunk)
		a.append('dependency_rel_head_word:\t'+self.dependency_rel_head_word)
		return '\n'.join(a)

	def set_info(self):
		'''Set POS information based on the output of Frog.'''
		if type(self.pos_line) == str:
			self.pos_line = self.pos_line.split('\t')

		pl = self.pos_line
		self.token_number = pl[0]
		self.token = pl[1]
		self.lemma = pl[2]
		self.morphological_segmentation = pl[3]
		self.pos = pl[4]
		self.pos_simple = self.pos.split('(')[0]
		self.pos_tag = pos2engpos[self.pos_simple]
		self.content_word = self.pos_simple in ['N','BW','WW','ADJ']
		self.probability_of_tag = pl[5]
		self.named_entity = pl[6]
		self.base_phrase_chunk = pl[7]
		self.token_number_of_head_word_in_dp_graph = pl[8]
		self.dependency_rel_head_word = pl[9]


pos2engpos = {
	'ADJ':'adjective',
	'BW':'Adverb',
	'LET':'Punctuation',
	'LID':'Determiner',
	'N':'Noun',
	'SPEC':'Names & unknown',
	'TSW':'Interjection',
	'TW':'Numerator',
	'VG':'Conjuction',
	'VNW':'Pronoun',
	'VZ':'Preposition',
	'WW':'Verb'
}
