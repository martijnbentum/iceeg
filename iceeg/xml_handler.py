import bad_channel
import bad_epoch
import combine_artifacts
from lxml import etree
import os
import path
import time

class xml_handler:
	def __init__(self,bad_epochs = [],bad_channels = [],bads = [],filename = '',multiplier = 1,artifact_type = 'bad_epoch'):
		'''Writes artifact info generated with manual_artifact_coder to xml files

		bad_epochs 	a list of bad_epoch objects, can be empty
		filename 	xml filename, for loading or writing
		'''
		self.bad_epochs = bad_epochs
		self.bad_channels = bad_channels
		self.bads = bads
		self.artifact_type = artifact_type
		self.filename = filename
		self.artifacts = etree.Element('artifacts')


	def __str__(self):
		return etree.tostring(self.artifacts, pretty_print=True).decode()


	def make_time(self):
		self.cdate = time.time()
		self.date = time.strftime("%b-%d-%Y-%H-%M-%S", time.localtime(self.cdate))

	def make_bads2xml(self):
		self.bad_epochs = self.bads.bads
		self.bad_channels = self.bads.bad_channels
		# self.bad_epochs2xml()
		# self.bad_channels2xml()
		self.bads2xml()
		info_xml = etree.SubElement(self.artifacts, 'info', id = 'bads-info')
		elements = 'name,nbe,nbc,ch_threshold,usability,block_duration,clean_duration,artifact_duration,clean_perc,artifact_perc,artifact_channels,be_annotations,bc_annotations,remove_ch'.split(',')
		for e in elements:
			element = etree.SubElement(info_xml,e)
			if not hasattr(self.bads,e):
				element.text = 'NA'
			elif e == 'remove_ch':
				element.text = ' '.join(getattr(self.bads,e))
			elif e in 'artifact_channels,be_annotations,bc_annotations'.split(','):
				d = getattr(self.bads,e)
				t = ' '.join([key + ',' + str(d[key]) for key in d])
				element.text = t
			else:
				element.text = str(getattr(self.bads,e))
		element = etree.SubElement(info_xml,'note')
		element.text = '''
		bad-epochs are stretches of eeg materials that contain artifacts or clean materials
		bad-epochs with artifact annotations refer to sections of eeg materials 
		that should be removed

		the bad-epochs in this file are based on manually corrected automatically generated
		bad-epoch and a subset of bad-channel annotations.

		the subset of bad-channels was determined as follows:
		if the sum of bad-channel artifact durations for a specific channel is shorter
		than 40% of the block duration they are part of the subset, otherwise the channel
		is added to the remove channel list

		remove_ch contains a list of channels that should be removed
		these channels show artifacts for more than 40% of the block duration
		Channel Fp2 is by default in the remove_ch list
		'''


	def bads2xml(self, multiplier = 1):
		'''Adds bad epochs from the m object to the xml tree.'''
		if self.artifact_type == 'bads' and hasattr(self,'artifacts'): pass
		else: self.artifacts = etree.Element('artifacts')
		for i,be in enumerate(self.bads.bads):
			if not be.ok:
				pass
			be_xml = etree.SubElement(self.artifacts, 'bad_epoch', id = str(i))
			# set epoch info elements
			elements = 'st_sample,et_sample,duration,block_st_sample,block_et_sample,pp_id,exp_type,bid,annotation'.split(',')
			sample_info = be.get_sample_info(multiplier = multiplier)
			for e in elements:
				# print(e)
				element = etree.SubElement(be_xml, e)
				if e == 'st_sample' and sample_info: element.text = str(sample_info[0])
				elif e == 'et_sample' and sample_info: element.text = str(sample_info[1])
				elif e == 'duration' and sample_info: element.text = str(sample_info[2])
				elif hasattr(be,e):
					element.text = str(getattr(be,e))
				else:
					element.text = 'NA'
		

	def bad_channels2xml(self, multiplier = 1):
		'''Adds bad epochs from the m object to the xml tree.'''
		if self.artifact_type == 'bads' and hasattr(self,'artifacts'): pass
		else: self.artifacts = etree.Element('artifacts')
		for bc in self.bad_channels:
			if not bc.ok:
				pass
			bc_xml = etree.SubElement(self.artifacts, 'bad_channel', id = str(bc.epoch_id))
			# set epoch info elements
			elements = 'channel,st_sample,et_sample,duration,block_st_sample,block_et_sample,pp_id,exp_type,bid,color,coder,correct,annotation,epoch_ids'.split(',')
			sample_info = bc.get_sample_info(multiplier = multiplier)
			for e in elements:
				# print(e)
				element = etree.SubElement(bc_xml, e)
				if e == 'st_sample' and sample_info: element.text = str(sample_info[0])
				elif e == 'et_sample' and sample_info: element.text = str(sample_info[1])
				elif e == 'duration' and sample_info: element.text = str(sample_info[2])
				elif hasattr(bc,e):
					element.text = str(getattr(bc,e))
				else:
					element.text = 'NA'

	def bad_epochs2xml(self, multiplier = 1):
		'''Adds bad epochs from the m object to the xml tree.'''
		if self.artifact_type == 'bads' and hasattr(self,'artifacts'): pass
		else: self.artifacts = etree.Element('artifacts')
		for be in self.bad_epochs:
			if not be.ok:
				pass
			be_xml = etree.SubElement(self.artifacts, 'bad_epoch', id = str(be.epoch_id))
			# set epoch info elements
			elements = 'st_sample,et_sample,duration,block_st_sample,block_et_sample,pp_id,exp_type,bid,color,coder,correct,annotation,epoch_ids'.split(',')
			sample_info = be.get_sample_info(multiplier = multiplier)
			for e in elements:
				# print(e)
				element = etree.SubElement(be_xml, e)
				if e == 'st_sample' and sample_info: element.text = str(sample_info[0])
				elif e == 'et_sample' and sample_info: element.text = str(sample_info[1])
				elif e == 'duration' and sample_info: element.text = str(sample_info[2])
				elif hasattr(be,e):
					element.text = str(getattr(be,e))
				else:
					element.text = 'NA'


	def load_xml(self, filename = None):
		'''Load xml data from file.'''
		if filename: self.filename = filename
		if not os.path.isfile(self.filename):
			print('Filename:',self.filename,'not found')
			return 0
		self.artifacts = etree.fromstring(open(self.filename).read())

	def xml2bad_channels(self,load_data = True, multiplier = 1):
		'''Create a list of bad epochs from xml file.'''
		self.bad_channels= []
		if load_data: self.load_xml()
		for bc_xml in self.artifacts.iter('bad_channel'):
			# fetch subelements
			element_names = 'channel,st_sample,et_sample,block_st_sample,block_et_sample,pp_id,bid,annotation,color,exp_type,epoch_ids,coder,correct'.split(',')
			element_values = []
			for e in element_names:
				if not bc_xml.find(e) == None: element_values.append(bc_xml.find(e).text)
				else: element_values.append('NA')
			channel,st_sample,et_sample,block_st_sample,block_et_sample,pp_id,bid,annotation,color,exp_type,epoch_ids,coder,correct= element_values
			if st_sample == 'NA' or et_sample == 'NA':
				continue
			epoch_id = bc_xml.attrib['id']
			#create start and end boundary
			# print(st_sample,et_sample)
			start = bad_epoch.Boundary(x = int(int(st_sample) * multiplier),boundary_type='start',visible = False)
			end = bad_epoch.Boundary(x = int(int(et_sample) * multiplier),boundary_type='end',visible = False)
			# create bad epoch
			bc = bad_channel.Bad_channel(channel,start_boundary = start, end_boundary = end, annotation = annotation, pp_id = pp_id, exp_type = exp_type, bid = bid,block_st_sample = block_st_sample,epoch_id = epoch_id, visible = False, epoch_ids = epoch_ids ,block_et_sample = block_et_sample,coder = coder,correct = correct,color = color)
			self.bad_channels.append(bc)
		# print('N bad channels:',len(self.bad_channels))
		return self.bad_channels


	def xml2bad_epochs(self,load_data = True, multiplier = 1, remove_clean = False):
		'''Create a list of bad epochs from xml file.'''
		self.bad_epochs = []
		if load_data: self.load_xml()
		for be_xml in self.artifacts.iter('bad_epoch'):
			# fetch subelements
			element_names = 'st_sample,et_sample,block_st_sample,block_et_sample,pp_id,bid,annotation,color,exp_type,coder,epoch_ids,coder,correct'.split(',')
			element_values = []
			for e in element_names:
				if not be_xml.find(e) == None: element_values.append(be_xml.find(e).text)
				else: element_values.append('NA')
			st_sample,et_sample,block_st_sample,block_et_sample,pp_id,bid,annotation,color,exp_type,coder,epoch_ids,coder,correct= element_values
			if remove_clean and annotation == 'clean': continue
			if st_sample == 'NA' or et_sample == 'NA':
				continue
			epoch_id = be_xml.attrib['id']
			#create start and end boundary
			# print(st_sample,et_sample)
			start = bad_epoch.Boundary(x = int(int(st_sample) * multiplier),boundary_type='start',visible = False)
			end = bad_epoch.Boundary(x = int(int(et_sample) * multiplier),boundary_type='end',visible = False)
			# create bad epoch
			be = bad_epoch.Bad_epoch(start_boundary = start, end_boundary = end, annotation = annotation, color = color,pp_id = pp_id, exp_type = exp_type, bid = bid,block_st_sample = block_st_sample,epoch_id = epoch_id, visible = False, epoch_ids = epoch_ids ,block_et_sample = block_et_sample,coder = coder,correct = correct)
			self.bad_epochs.append(be)
		# print('N bad epoch:',len(self.bad_epochs))
		return self.bad_epochs


	def xml2bads(self,load_data = True, remove_clean = False):
		if load_data: self.load_xml()
		self.bad_epochs = []
		self.bad_channels = []
		self.xml2bad_epochs(load_data = False,remove_clean = remove_clean)
		self.xml2bad_channels(load_data = False)
		elements = 'name,nbe,nbc,ch_threshold,usability,block_duration,clean_duration,artifact_duration,clean_perc,artifact_perc,artifact_channels,be_annotations,bc_annotations,remove_ch'.split(',')
		info_xml = self.artifacts.find('info')
		element_values = []
		bads = combine_artifacts.bads(name = '-')
		for e in elements:
			t = info_xml.find(e) 
			if not t == None:
				t = t.text
				if e == 'remove_ch':
					setattr(bads,e,t.split(' '))
				elif e in 'artifact_channels,be_annotations,bc_annotations'.split(','):
					d = dict([line.split(',') for line in t.split(' ')])
					setattr(bads,e,d)
				elif e in 'nbe,nbc,block_duration,clean_duration,artifact_duration'.split(','): 
					setattr(bads,e,int(t))
				elif e in 'ch_threshold,clean_perc,artifact_perc'.split(','):
					setattr(bads,e,float(t))
				else:
					setattr(bads,e,t)
			else: setattr(bads,e,'NA')
		bads.bad_channels = self.bad_channels
		bads.bads = self.bad_epochs
		self.bads = bads
		return bads


	def write(self):
		'''Writes data to xml file and moves a copy of the previous version to folder OLD'''
		self.make_time()
		if 'channel' in self.filename: bads = self.bad_channels
		else: bads = self.bad_channels
		if os.path.isfile(self.filename): 
			directory = '/'.join(self.filename.split('/')[:-1])
			print('moving file:','mv ' + self.filename +  ' ' + directory + '/OLD/' + self.filename.split('/')[-1].strip('.xml') + '_'+self.date + '_nbad_epochs-'+ str(len(bads)) + '.xml')
			os.system('mv ' + self.filename +  ' ' + directory+ '/OLD/' + self.filename.split('/')[-1].rstrip('.xml') + '_'+self.date + '_nbad_epochs-'+ str(len(bads)) + '.xml')
		self.save(self.filename)

	def save(self, filename):
		print('saving xml file to:',filename)
		fout = open(filename,'w')
		fout.write(etree.tostring(self.artifacts, pretty_print=True).decode())
		fout.close()


	
