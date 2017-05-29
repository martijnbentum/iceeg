import difflib

class Matcher:
	def __init__(self,ort,awd):
		# uses difflib to match ort and awd words. ort can be str or list, awd must be list
		self.set_values(ort,awd) # set a str and wl version for both ort & awd
		# self.fix_ort() # check for mistakes in ort transcription, interpunction
		self.nort = len(self.wl_ort) 
		self.nawd = len(self.wl_awd)
		self.nequal = self.nort == self.nawd
		self.match = difflib.SequenceMatcher(a=self.wl_ort,b=self.wl_awd) # match word lists
		self.blocks = self.match.get_matching_blocks() # get matching blocks of indices
		self.ratio = self.match.ratio() # check percentage of items that correspond
		self.make_translation() # create dicts that translate indices of ort to wl and vice versa
		self.check_problems() # checks in what way if any awd and ort are different
		self.complete_match = self.ratio == 1


	def __str__(self):
		a = ['ort:\t\t\t'+self.s_ort]
		a.append('awd:\t\t\t'+self.s_awd)
		a.append('equal:\t\t\t'+str(self.nequal))
		a.append('asr problem:\t\t'+str(self.asr_problem))
		a.append('ort index missing:\t'+' '.join(map(str,self.ort_indices_missing_in_awd)))
		a.append('awd index missing:\t'+' '.join(map(str,self.awd_indices_missing_in_ort)))
		a.append('no words match:\t\t'+ str(self.no_match))
		a.append('match ratio:\t\t'+ str(self.ratio))
		a.append('complete match:\t\t'+ str(self.complete_match))
		a.append('awd wl:\t\t\t'+'   -   '.join(self.wl_awd))
		a.append('ort wl:\t\t\t'+'   -   '.join(self.wl_ort))
		return '\n'.join(a)

	def print_fields(self):
		# prints all fieldnames of the object
		a = 's_ort,s_awd,wl_ort,wl_awd,nort,nawd,nequal,match,blocks,ratio,asr_problem,ort_indices_missing_in_awd,awd_indices_missing_in_ort,no_match,ort_index2awd_index,awd_index2ort_index,complete_match'.split(',')
		print('\n'.join(a))
		return a

	def set_values(self,ort,awd):
		# sets ort and awd to a string and list variables
		removables = ['\\','.',',','?']
		if type(ort) == list:
			ort = ' '.join(ort)

		assert type(ort) == str
		for r in removables:
			ort = ort.replace(r,'')
		self.wl_ort = ort.split(' ')
		self.s_ort = ort

		assert type(awd) == list
		for r in removables:
			awd = [w.replace(r,'') for w in awd if w != '']
		self.s_awd = ' '.join(awd)
		self.wl_awd = awd
 
	def check_problems(self):
		# tries to diagnose alignment problems between ort and awd words
		self.asr_problem = False
		self.ort_indices_missing_in_awd = []
		self.awd_indices_missing_in_ort = []
		self.no_match = False

		if self.nequal == False:
			for awd_word in self.wl_awd:
				# sometimes the words are not split in the ASR alignment
				if ' ' in awd_word: self.asr_problem = True
		
		self.ort_all_indices = list(range(len(self.wl_ort)))
		self.awd_all_indices = list(range(len(self.wl_awd)))
		self.ort_indices_missing_in_awd = [i for i in self.ort_all_indices if i not in self.ort_index2awd_index.keys()]
		self.awd_indices_missing_in_ort = [i for i in self.awd_all_indices if i not in self.awd_index2ort_index.keys()]

		if len(self.ort_index2awd_index) == 0: self.no_match = True
				
		
	def make_translation(self):
		# creates a dict that mapes ort word indices to awd word indices
		self.ort_index2awd_index = {} 
		self.awd_index2ort_index = {} 
		for block in self.blocks:
			if block.size == 0:
				pass
			indices_ort = list(range(block.a,block.a + block.size))
			indices_awd = list(range(block.b,block.b + block.size))
			for i,i_ort in enumerate(indices_ort):
				self.ort_index2awd_index[i_ort] = indices_awd[i]
				self.awd_index2ort_index[indices_awd[i]] = i_ort
			
