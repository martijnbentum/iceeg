import bad_epoch
from lxml import etree
import os
import time

class xlm_handler:
	def __init__(self,bad_epochs = [],filename = '',coder_name = '',cdate = None, data_type = 'learning'):
		'''Writes artifact info generated with manual_artifact_coder to xml files

		bad_epochs 	a list of bad_epoch objects, can be empty
		filename 	xml filename, for loading or writing
		cdate 		epoch time (time.time()), default is init time
		data_type 	learning test result, default is learning
		'''
		self.bad_epochs = bad_epochs
		self.filename = filename
		if cdate == None: cdate = time.time()
		date = time.strftime("%b %d %Y %H:%M:%S", time.localtime(cdate))
		self.artifacts = etree.Element('artifacts')


	def __str__(self):
		return etree.tostring(self.artifacts, pretty_print=True).decode()


	def bad_epochs2xml(self):
		'''Adds bad epochs from the m object to the xml tree.'''
		for be in self.bad_epochs:
			if not be.ok:
				pass
			be_xml = etree.SubElement(self.artifacts, 'bad_epoch', id = str(be.epoch_id))
			# set epoch info elements
			elements = 'st_sample,et_sample,duration,block_st_sample,pp_id,exp_type,bid,color,annotation'.split(',')
			for e in elements:
				element = etree.SubElement(be_xml, e)
				if hasattr(be,e):
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


	def xml2bad_epochs(self,load_data = True):
		'''Create a list of bad epochs from xml file.'''
		if load_data: self.load_xml()
		else: print('Starting with list of',len(artifacts,' bad epochs'))
		for be_xml in self.artifacts.iter('bad_epoch'):
			# fetch subelements
			element_names = 'st_sample,et_sample,block_st_sample,pp_id,bid,annotation,color,exp_type'.split(',')
			st_sample,et_sample,block_st_sample,pp_id,bid,annotation,color,exp_type= [ be_xml.find(e).text for e in element_names]
			epoch_id = be_xml.attrib['id']
			#create start and end boundary
			start = bad_epoch.Boundary(int(st_sample),'start')
			end = bad_epoch.Boundary(int(et_sample),'end')
			# create bad epoch
			be = bad_epoch.Bad_epoch(start_boundary = start, end_boundary = end, annotation = annotation, color = color,pp_id = pp_id, exp_type = exp_type, bid = bid,block_st_sample = block_st_sample,epoch_id = epoch_id, visible = False )
			self.bad_epochs.append(be)
		print('N bad epoch:',len(self.artifacts),'artifacts')


	def write(self):
		'''Writes data to xml file.'''
		fout = open(self.filename,'w')
		fout.write(etree.tostring(self.artifacts, pretty_print=True).decode())
		fout.close()
