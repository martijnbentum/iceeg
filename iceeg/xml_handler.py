import bad_channel
import bad_epoch
from lxml import etree
import os
import path
import time

class xml_handler:
	def __init__(self,bad_epochs = [],bad_channels = [],filename = '',multiplier = 1,artifact_type = 'bad_epoch'):
		'''Writes artifact info generated with manual_artifact_coder to xml files

		bad_epochs 	a list of bad_epoch objects, can be empty
		filename 	xml filename, for loading or writing
		'''
		self.bad_epochs = bad_epochs
		self.bad_channels = bad_channels
		self.filename = filename
		self.artifacts = etree.Element('artifacts')


	def __str__(self):
		return etree.tostring(self.artifacts, pretty_print=True).decode()


	def make_time(self):
		self.cdate = time.time()
		self.date = time.strftime("%b-%d-%Y-%H-%M-%S", time.localtime(self.cdate))

	def bad_channels2xml(self, multiplier = 1):
		'''Adds bad epochs from the m object to the xml tree.'''
		self.artifacts = etree.Element('artifacts')
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
		self.artifacts = etree.Element('artifacts')
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
			element_names = 'channel,st_sample,et_sample,block_st_sample,block_et_sample,pp_id,bid,annotation,color,exp_type,coder,epoch_ids,coder'.split(',')
			element_values = []
			for e in element_names:
				if not bc_xml.find(e) == None: element_values.append(bc_xml.find(e).text)
				else: element_values.append('NA')
			channel,st_sample,et_sample,block_st_sample,block_et_sample,pp_id,bid,annotation,color,exp_type,coder,epoch_ids,coder= element_values
			if st_sample == 'NA' or et_sample == 'NA':
				continue
			epoch_id = bc_xml.attrib['id']
			#create start and end boundary
			print(st_sample,et_sample)
			start = bad_epoch.Boundary(x = int(int(st_sample) * multiplier),boundary_type='start',visible = False)
			end = bad_epoch.Boundary(x = int(int(et_sample) * multiplier),boundary_type='end',visible = False)
			# create bad epoch
			bc = bad_channel.Bad_channel(channel,start_boundary = start, end_boundary = end, annotation = annotation, pp_id = pp_id, exp_type = exp_type, bid = bid,block_st_sample = block_st_sample,epoch_id = epoch_id, visible = False, epoch_ids = epoch_ids ,block_et_sample = block_et_sample,coder = coder)
			self.bad_channels.append(bc)
		print('N bad channels:',len(self.bad_channels))
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
			print(st_sample,et_sample)
			start = bad_epoch.Boundary(x = int(int(st_sample) * multiplier),boundary_type='start',visible = False)
			end = bad_epoch.Boundary(x = int(int(et_sample) * multiplier),boundary_type='end',visible = False)
			# create bad epoch
			be = bad_epoch.Bad_epoch(start_boundary = start, end_boundary = end, annotation = annotation, color = color,pp_id = pp_id, exp_type = exp_type, bid = bid,block_st_sample = block_st_sample,epoch_id = epoch_id, visible = False, epoch_ids = epoch_ids ,block_et_sample = block_et_sample,coder = coder,correct = correct)
			self.bad_epochs.append(be)
		print('N bad epoch:',len(self.bad_epochs))
		return self.bad_epochs


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


	
