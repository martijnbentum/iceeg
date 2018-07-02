from matplotlib import pyplot as plt
import copy
from lxml import etree
import os
import path
import time
import windower

class eog:
	'''stores eog information
	'''
		
	def __init__(self,scores=[],comps=[],b = None,s = None,bg = [],name = '',filename = '', coder = 'martijn', ica_filename = '',ica_corrected = 'no',identifier = '_corrected_no-artifact'):
		'''Information about stretch of eeg data containing artefacts.
		scores 				correlation scores of each component and veog and heog channels
		comps 				components selected by mne eog detector
		b 					block
		s 					session
		bg 					block_group
		filename 			xml filename
		'''
		self.scores = [scores[0].round(3),scores[1].round(3)]
		try: self.veog_scores_all, self.heog_scores_all = self.scores
		except: self.veog_scores_all, self.heog_scores_all = [], []
		self.comps = comps
		self.b = b
		self.s = s
		self.bg = bg
		self.name = name
		self.filename = filename
		self.coder = coder
		self.ica_filename = ica_filename
		self.ica_corrected = ica_corrected
		self.identifier = identifier
		self.added_comps = []
		self.deleted_comps = []
		self.set_comps()
		if b and not filename: self.set_filename()

	def __str__(self):
		m = 'Name:\t\t\t'+self.name + '\n'
		m += 'filename:\t\t'+ self.filename + '\n'
		m += 'ica_filename:\t\t'+ self.ica_filename + '\n'
		if len(self.scores) > 0 and len(self.scores[0]) > 0:
			m += 'components:\t\t'+' '.join(map(str,self.comps)) + '\n'
			m += 'veog comp:\t\t'+' '.join(map(str,self.veog_comps_selection)) + '\n'
			m += 'heog comp:\t\t'+' '.join(map(str,self.heog_comps_selection)) + '\n'
			m += 'veog scores:\t\t'+' '.join(map(str,self.scores[0][self.comps])) + '\n'
			m += 'heog scores:\t\t'+' '.join(map(str,self.scores[1][self.comps])) + '\n'
		else: m += 'no components correlate with eog.\n'
		m += 'added comps: \t\t'+ ' '.join(map(str,self.added_comps)) + '\n'
		m += 'deleted comps: \t\t'+ ' '.join(map(str,self.deleted_comps)) + '\n'
		return m

	def __repr__(self):
		m = 'eog object\t\t' + self.comps_string()
		return m
			
	def set_filename(self):
		self.name = windower.make_name(self.b)
		self.filename = self.name + self.identifier + '-eog.xml'

	def set_comps(self):
		'''Set heog and veog components and scores '''
		if len(self.scores) != 2: return False
		scores = self.scores
		self.veog_comps, self.heog_comps= [], []
		self.veog_scores, self.heog_scores= [], []
		for i in range(len(scores[0])):
			if i in self.comps:
				if abs(scores[0][i]) > abs(scores[1][i]):
					self.veog_comps.append(i)
					self.veog_scores.append(scores[0][i])
				else:
					self.heog_comps.append(i)
					self.heog_scores.append(scores[1][i])
		self.ncomps = len(self.comps)


	def add_comp(self,component_index):
		self.added_comps.append(component_index)
		self.comps.append(component_index)
		self.set_comps()

	
	def delete_comp(self,component_index):
		if component_index in self.comps:
			self.comps.pop(self.comps.index(component_index))
			self.deleted_comps.append(component_index)
		self.set_comps() 


	def comps_string(self):
		return ','.join(map(str,self.comps))


	def scores_string(self):
		if self.veog_scores_all != [] and self.heog_scores_all != []:
			veog = ','.join(map(str,self.veog_scores_all))
			heog = ','.join(map(str,self.heog_scores_all))
			self.veog_score_str = veog
			self.heog_score_str = heog
			return '\t'.join([veog,heog])
		else:
			print('no scores to stringify, returning empty string.')
			return ''


	def eog2xml(self):
		'''Creates xml version of eog object'''
		self.eog_xml= etree.Element('eog')
		# set epoch info elements
		elements = 'name,comps,scores,heog_comps,veog_comps,heog_scores,veog_scores,deleted_comps,added_comps,ica_filename,filename,coder,ica_corrected,identifier,ncomps'.split(',')
		for e in elements:
			# print(e)
			element = etree.SubElement(self.eog_xml, e)
			if e == 'comps': element.text = self.comps_string()
			elif e == 'scores': element.text = self.scores_string()
			elif hasattr(self,e):
				attr = getattr(self,e)
				if type(attr) == list: attr = ','.join(map(str,attr))
				else: attr = str(attr)
				if attr == '': element.text = 'NA'
				element.text = attr
			else:
				element.text = 'na'


	def set_lists(self,list_values):
		comps,scores,heog_comps,veog_comps,heog_scores,veog_scores,deleted_comps,added_comps= list_values 
		self.comps = list(map(int,comps.split(',')))
		veog, heog = scores.split('\t')
		self.scores = [np.array(veog.split(',')), np.array(heog.split(','))]
		self.veog_comps = list(map(int,veog_comps.split(',')))
		self.heog_comps = list(map(int,heog_comps.split(',')))
		self.added_comps = list(map(int,added_comps.split(',')))
		self.deleted_comps = list(map(int,deleted_comps.split(',')))
		self.veog_scores =list(map(int,veog_scores.split(',')))
		self.heog_scores = list(map(int,heog_scores.split(',')))

		try: self.ncomps = int(self.ncomps)
		except: pass


	def xml2eog(self):
		'''Create a list of bad epochs from xml file.'''
		if not hasattr(self,'eog_xml'): 
			if not self.load(): return False
			# fetch subelements
		list_elements = 'comps,scores,heog_comps,veog_comps,heog_scores,veog_scores,deleted_comps,added_comps'.split(',')
		other_elements = 'name,ica_filename,filename,coder,ica_corrected,identifier,ncomps'.split(',')
		list_values = []
		all_elements = list_elements + other_elements
		for e in all_elements:
			value = self.eog_xml.find(e)
			if not value == None: text = 'NA'
			else: text = value.text

			if e in other_elements: setattr(self,e,text)
			else:list_values.append(text)
		self.set_lists(self,list_values)


	def write(self):
		self.eog2xml()
		print('saving xml file to:',self.filename)
		fout = open(path.ica_solutions + 'EOG/' + self.filename,'w')
		fout.write(etree.tostring(self.eog_xml, pretty_print=True).decode())
		fout.close()

	def load(self):
		if not hasattr(self,'filename'): return False
		self.eog_xml = etree.fromstring(open(path.ica_solutions + 'EOG/' +self.filename).read())
		self.xml2eog()


def load(filename):
	e = eog(filename = filename)
	e.load()
	return e
	

