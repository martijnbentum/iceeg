from lxml import etree
import notes
import os
import path
import time

class xml_notes:
	'''Create xml from note object or create an object from xml.'''
	def __init__(self,note = None,filename = ''):
		'''Writes note to xml files, should provide note or filename.

		note 		note object (notes.py)
		filename 	xml filename, for loading or writing
		'''
		self.note= note
		if note != None: self.filename = note.filename
		else: self.filename = filename
		self.notes= etree.Element('notes')


	def __str__(self):
		return etree.tostring(self.notes, pretty_print=True).decode()


	def make_time(self):
		'''Create current time to add to filename of previous versions in OLD folder.'''
		self.cdate = time.time()
		self.date = time.strftime("%b-%d-%Y-%H-%M-%S", time.localtime(self.cdate))

	def note2xml(self):
		'''Adds note to the xml tree.
		The note has the following childer: channel_notes, general_notes and log
		log gets a child each time the note is updated (saved again), previous text is 
		preserved the last log is the text that is displayed in the note field on the note object.
		'''
		self.artifacts = etree.Element('notes')
		note_xml = etree.SubElement(self.notes, 'note')
		ch_xml = etree.SubElement(note_xml, 'channel_notes')
		general_xml = etree.SubElement(note_xml, 'general_notes')
		log_xml = etree.SubElement(note_xml, 'log')
		for key in self.note.old_logs.keys():
			l = etree.SubElement(log_xml,key)
			l.text = self.note.old_logs[key]
		log = etree.SubElement(log_xml,self.note.coder + '_' + self.note.last_edited.replace(':','-').replace(' ','_'))
		log.text = self.note.note
		for e in self.note.general_notes.keys():
			element = etree.SubElement(general_xml, e)
			element.text = self.note.general_notes[e]
		for e in self.note.ch_notes.keys():
			element = etree.SubElement(ch_xml, e)
			element.text = self.note.ch_notes[e]

		elements = 'name,annotation_type,note_type,created,last_edited,coder'.split(',')
		for e in elements:
			element = etree.SubElement(note_xml, e)
			element.text = self.note.name
			element.text = str(getattr(self.note,e))


	def load_xml(self, filename = None):
		'''Load xml data from file.'''
		if filename: self.filename = filename
		if not os.path.isfile(self.filename):
			print('Filename:',self.filename,'not found')
			return 0
		self.notes = etree.fromstring(open(self.filename).read())

	def xml2note(self,load_data = True):
		'''Create a note from a xml file.'''
		if load_data: self.load_xml()

		note_xml = self.notes.find('note')
		ch_xml = note_xml.find('channel_notes')
		general_xml = note_xml.find('general_notes')
		text_xml = note_xml.find('log')

		elements =ch_xml.getchildren()
		ch_dict = {} 
		for e in elements:
			ch_dict[e.tag] = e.text
		elements =general_xml.getchildren()
		general_dict = {}
		for e in elements:
			general_dict[e.tag] = e.text

		log_xml= note_xml.find('log')
		elements =log_xml.getchildren()
		logs_dict = {}
		for i,e in enumerate(elements):
			if i == len(elements) -1:
				note = e.text
			logs_dict[e.tag] = e.text

		elements = 'name,annotation_type,note_type,created,last_edited,coder'.split(',')
		element_values = []
		for e in elements:
			print(e)
			if note_xml.find(e).text:
				element_values.append(note_xml.find(e).text)
			else: element_values.append('')
		name,annotation_type,note_type,created,last_edited,coder= element_values

		self.note = notes.note(name = name,note= note,ch_notes = ch_dict,general_notes=general_dict,created=created,last_edited =last_edited,coder = coder,annotation_type=annotation_type,note_type=note_type,old_logs = logs_dict,try_load = False)


	def write(self):
		'''Writes data to xml file and moves a copy of the previous version to folder OLD'''
		self.make_time()
		if os.path.isfile(self.filename): 
			directory = '/'.join(self.filename.split('/')[:-1])
			print('moving file:','mv ' + self.filename +  ' ' + directory + '/OLD/' + self.filename.split('/')[-1].strip('.xml') + '_'+self.date +  '.xml')
			os.system('mv ' + self.filename +  ' ' + directory+ '/OLD/' + self.filename.split('/')[-1].rstrip('.xml') + '_'+self.date + '.xml')
		self.save(self.filename)

	def save(self, filename):
		print('saving xml file to:',filename)
		fout = open(filename,'w')
		fout.write(etree.tostring(self.notes, pretty_print=True).decode())
		fout.close()


	
