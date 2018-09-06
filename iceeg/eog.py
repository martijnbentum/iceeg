from matplotlib import pyplot as plt
import copy
from lxml import etree
import numpy as np
import os
import path
import time
import windower

class eog:
	'''stores eog information
	'''
		
	def __init__(self,scores=[],comps=[],b = None,s = None,bg = [],name = '',filename = '', coder = 'martijn', ica_filename = '',ica_checked = False,identifier = '_corrected_no-artifact',loaded_bid = [],rejected_bid = [], not_loaded_bid = [], rejected_channels = []):
		'''Information about stretch of eeg data containing artefacts.
		scores 				correlation scores of each component and veog and heog channels
		comps 				components selected by mne eog detector
		b 					block
		s 					session
		bg 					block_group
		filename 			xml filename
		'''
		try:
			self.scores = [scores[0].round(3),scores[1].round(3)]
			self.veog_scores_all, self.heog_scores_all = self.scores
		except: self.veog_scores_all, self.heog_scores_all = [], []
		self.comps = comps
		self.b = b
		self.s = s
		self.bg = bg
		self.name = name
		self.filename = remove_dir(filename)
		self.coder = coder
		self.ica_filename = remove_dir(ica_filename)
		self.ica_checked= ica_checked
		self.identifier = identifier
		self.added_comps = []
		self.deleted_comps = []
		self.loaded_bid = loaded_bid
		self.rejected_bid = rejected_bid
		self.not_loaded_bid = not_loaded_bid
		self.rejected_channels = rejected_channels
		if hasattr(self,'scores'): self.set_comps()
		if (b or s) and not filename: self.set_filename()

	def __str__(self):
		m = 'Name:\t\t'+self.name + '\n'
		m += 'filename:\t'+ self.filename + '\n'
		m += 'ica_filename:\t'+ self.ica_filename + '\n'
		if len(self.scores) > 0 and len(self.scores[0]) > 0:
			m += 'components:\t'+' '.join(map(str,self.comps)) + '\n'
			m += 'veog comp:\t'+' '.join(map(str,self.veog_comps)) + '\n'
			m += 'heog comp:\t'+' '.join(map(str,self.heog_comps)) + '\n'
			m += 'veog scores:\t'+' '.join(map(str,self.scores[0][self.comps])) + '\n'
			m += 'heog scores:\t'+' '.join(map(str,self.scores[1][self.comps])) + '\n'
		else: m += 'no components correlate with eog.\n'
		m += 'added comps: \t'+ ' '.join(map(str,self.added_comps)) + '\n'
		m += 'deleted comps: \t'+ ' '.join(map(str,self.deleted_comps)) + '\n'
		return m

	def __repr__(self):
		m = 'eog object\t\t' + self.comps_string()
		return m
			
	def set_filename(self):
		if self.b:
			self.name = windower.make_name(self.b)
		if self.s:
			self.name = self.s.name
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
		if component_index in self.comps:
			print('already selected, doing nothing')
			return False
		self.added_comps.append(component_index)
		self.comps.append(component_index)
		self.set_comps()
		return True

	
	def delete_comp(self,component_index):
		if component_index in self.comps:
			self.comps.pop(self.comps.index(component_index))
			if not component_index in self.added_comps:
				self.deleted_comps.append(component_index)
			else: self.added_comps.pop(self.added_comps.index(component_index))
			self.set_comps() 
			return True
		print('component not in selection, doing nothing')
		return False


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
		elements = 'name,comps,scores,heog_comps,veog_comps,heog_scores,veog_scores,deleted_comps,added_comps,ica_filename,filename,coder,ica_checked,identifier,ncomps,loaded_bid,rejected_bid,not_loaded_bid,rejected_channels'.split(',')
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
		comps,scores,heog_comps,veog_comps,heog_scores,veog_scores,deleted_comps,added_comps,loaded_bid,rejected_bid,not_loaded_bid,rejected_channels= list_values 
		try:self.loaded_bid = loaded_bid.split(',')
		except: self.loaded_bid = []
		try: self.not_loaded_bid = not_loaded_bid.split(',')
		except: self.not_loaded_bid = []
		try:self.rejected_channels = rejected_channels.split(',')
		except: self.rejected_channels= []
		if comps:self.comps = list(map(int,comps.split(',')))
		else:self.comps = []
		veog, heog = scores.split('\t')
		self.scores = [np.array(list(map(float,veog.split(',')))), np.array(list(map(float,heog.split(','))))]
		self.veog_scores_all = self.scores[0]
		self.heog_scores_all = self.scores[1]
		if veog_comps != None:
			self.veog_comps = list(map(int,veog_comps.split(',')))
			self.veog_scores =list(map(float,veog_scores.split(',')))
		else: self.veog_comps,self.veog_scores = [],[]
		if heog_comps != None:
			self.heog_comps = list(map(int,heog_comps.split(',')))
			self.heog_scores = list(map(float,heog_scores.split(',')))
		else: self.heog_comps,self.heog_scores = [],[]
		try:self.added_comps = list(map(int,added_comps.split(',')))
		except:self.added_comps = []
		try:self.deleted_comps = list(map(int,deleted_comps.split(',')))
		except:self.deleted_comps = []
		try: self.ncomps = int(self.ncomps)
		except: pass


	def xml2eog(self):
		'''Create a list of bad epochs from xml file.'''
		if not hasattr(self,'eog_xml'): 
			if not self.load(): return False
			# fetch subelements
		list_elements = 'comps,scores,heog_comps,veog_comps,heog_scores,veog_scores,deleted_comps,added_comps,loaded_bid,rejected_bid,not_loaded_bid,rejected_channels'.split(',')
		other_elements = 'name,ica_filename,filename,coder,ica_checked,identifier,ncomps'.split(',')
		list_values = []
		all_elements = list_elements + other_elements
		for e in all_elements:
			value = self.eog_xml.find(e)
			if value == None: text = 'NA'
			else: text = value.text

			if e in other_elements: setattr(self,e,text)
			else:list_values.append(text)
		self.set_lists(list_values)


	def write(self):
		self.eog2xml()
		print('saving xml file to:',self.filename)
		fout = open(path.ica_solutions + self.filename,'w')
		fout.write(etree.tostring(self.eog_xml, pretty_print=True).decode())
		fout.close()

	def load(self):
		if not hasattr(self,'filename'): return False
		self.eog_xml = etree.fromstring(open( path.ica_solutions + self.filename).read())
		self.xml2eog()
		
	def print_xml(self):
		return etree.tostring(self.eog_xml, pretty_print=True).decode()


def load(filename):
	if '/' not in filename: filename = path.ica_solutions + filename
	if not os.path.isfile(filename): 
		print('could not find file:',filename)
		return False
	e = eog(filename = filename)
	e.load()
	return e
	

def remove_dir(filename):
	if '/' in filename: return filename.split('/')[-1]
	else: return filename
	
