import bad_epoch
import check_artifact_channel_length as cacl
import check_artifact_length as cal
import copy
import glob
import notes
import os
import path
import utils
import xml_handler

class bads():
	'''Object to contain bad epoch and bad channels
	bad channels will be converted to bad epochs if they are bad les than ch_theshold of the block.
	'''
	def __init__(self, name, ch_threshold =.4, minimal_clean_duration = 2000, fo = None, force_make = False):
		'''Create bads object to hold annotation information and save this to xml.
		name 			name of block
		ch_threshold 	percentage of block bad for a channel to be completely removed
		minimal... 		the minimal time between artifacts (otherwise artifacts are combined)
		fo 				fid2ort object, speeds up loading of blocks
		force_make 		whether to overwrite existing bad xml
		'''
		if name == '-': return
		f = path.bads_annotations + 'bads_'+name+'.xml'
		self.epoch_id = 1
		self.name = name
		self.fo = fo
		self.force_make = force_make

		if not force_make and os.path.isfile(f):
			x = xml_handler.xml_handler(filename = f, artifact_type = 'bads')
			bads = x.xml2bads()
			self.__dict__.update(bads.__dict__)
		else:
			self.b = utils.name2block(name,fo)
			self.minimal_clean_duration = minimal_clean_duration
			self.read_all()
			self.annotations()
			self.block_duration = utils.load_100hz_numpy_block(name).shape[-1] * 10
			self.select_remove_ch(ch_threshold)
			self.make_bads()
			self.make_info()
			self.save()
		
		

	def __str__(self):
		m = 'name\t\t'+self.name +'\n'
		m += 'nbe\t\t'+str(self.nbe)+'\n'
		m += 'nbc\t\t'+str(self.nbc)+'\n'
		m += 'be_annot\t'+str(self.be_annotations)+'\n'
		m += 'bc_annot\t'+str(self.bc_annotations)+'\n'
		m += 'artifact ch\t' + '\n\t\t'.join([k+'\t'+str(v) for k,v in sorted(self.artifact_channels.items(), key=lambda x: x[1],reverse=True)]) + '\n'
		m += 'ch threshold\t'+str(self.ch_threshold)+'\n'
		m += 'remove ch\t'+' '.join(self.remove_ch)+'\n'
		m += 'usability\t'+ self.usability+'\n'
		m += 'block dur\t'+str(self.block_duration)+'\n'
		m += 'clean dur\t'+ str(self.clean_duration) + '\n'
		m += 'artifact dur\t'+ str(self.artifact_duration) + '\n'
		m += 'clean perc\t'+ str(round(self.clean_perc,3)) + '\n'
		m += 'artifact perc\t'+ str(round(self.artifact_perc,3)) + '\n'
		return m

	def __repr__(self):
		m = 'bads\t'+self.name 
		m += '\t' + self.usability
		m += '\tclean: ' + str(round(self.clean_perc,2))  
		m += '\trm ch: ' + ' '.join(self.remove_ch)
		return m

	def read_bad_epochs(self):
		'''read the bad epochs from file, should be the corrected automatic annotations.
		'''
		self.f_be = get_xml_filename(self.name,bad_type='bad_epoch')
		self.be = xml_handler.xml_handler(filename = self.f_be,artifact_type = 'bad_epoch')
		self.be.xml2bad_epochs()

	def read_bad_channels(self):
		'''read the bad channel from file, should be the corrected automatic annotations.
		'''
		self.f_bc = get_xml_filename(self.name,bad_type='bad_channel')
		self.bc = xml_handler.xml_handler(filename = self.f_bc,artifact_type = 'bad_channel')
		self.bc.xml2bad_channels()

	def read_note(self):
		'''Read the note of the block, stating the usability of the block.
		'''
		self.n = notes.note(self.name)

	def read_all(self):
		'''Read all xml files corresponding to this block.'''
		self.read_bad_epochs()
		self.read_bad_channels()
		self.read_note()

	def annotations(self):
		'''Create annotation dictionaries for epochs and channels to check which 
		annotations are used.
		'''
		self.bc_annotations = {}
		self.be_annotations = {}
		self.artifact_channels = {}
		for epoch in self.be.bad_epochs:
			if epoch.annotation not in self.be_annotations.keys():
				self.be_annotations[epoch.annotation] = epoch.duration
			else: self.be_annotations[epoch.annotation] += epoch.duration
		for epoch in self.bc.bad_channels:
			if epoch.annotation not in self.bc_annotations.keys():
				self.bc_annotations[epoch.annotation] = epoch.duration
			else: self.bc_annotations[epoch.annotation] += epoch.duration
			if epoch.channel not in self.artifact_channels.keys(): 
				self.artifact_channels[epoch.channel] = epoch.duration
			else:
				self.artifact_channels[epoch.channel] += epoch.duration
		self.bc_annot_labels = self.bc_annotations.keys()
		self.be_annot_labels = self.be_annotations.keys()
		self.bc_duration = 1

	def select_remove_ch(self, threshold = None):
		'''Create a set of channels that should be removed based on the ch_threshold.
		these should not be used to create bad epochs.
		'''
		if threshold != None: self.ch_threshold = threshold
		self.remove_ch = ['Fp2']
		self.ms_threshold = self.block_duration * self.ch_threshold
		for ch in self.artifact_channels:
			if self.artifact_channels[ch] > self.ms_threshold: self.remove_ch.append(ch)
		

	def make_bads(self,minimal_duration = None):
		'''Combine bads (epochs and channels into the format used for artifact clean files. 
		Clean sections should at least be minimal duration long.
		'''
		if minimal_duration == None: minimal_duration = self.minimal_clean_duration
		self.temp_bads = []
		artifact_channels = [ch for ch in self.bc.bad_channels if ch.channel not in self.remove_ch]
		self.bad_channels = artifact_channels[:]
		self.temp = artifact_channels + self.be.bad_epochs
		for b in self.temp:
			if b.annotation == 'clean': continue
			if b.annotation != 'artifact':print(b,'annotation: '+b.annotation,'could be unwanted, clean is ignored')
			start = bad_epoch.Boundary(b.st_sample,boundary_type = 'start',visible = False)
			end= bad_epoch.Boundary(b.et_sample,boundary_type = 'end',visible = False)
			be = bad_epoch.Bad_epoch(start,end,'artifact',b.coder,'blue',b.pp_id,b.exp_type,b.block_st_sample,b.bid,self.epoch_id,False,'correct',-9,'',b.block_st_sample + self.block_duration)
			self.temp_bads.append(be)
			self.epoch_id +=1
		self.temp_bads.sort()
		if len(self.temp_bads) == 0: 
			self.bads = []
			return

		temp_bads = copy.deepcopy(self.temp_bads)
		temp_bads = cal.combine_overlaps(temp_bads)
		self.stiches = cal.stitch_artifacts(temp_bads,minimal_duration) 
		self.stiched_stiches = cal.stitch_stiches(self.stiches)
		temp_bads=  cal.combine_artifacts(temp_bads,self.stiched_stiches)
		cal.check_artifacts(artifacts=temp_bads,fo = self.fo,default='clean',minimal_duration = minimal_duration)
		self.bads = temp_bads


	def make_info(self):
		'''Creates aggregate information for the block about channel and epoch artifacts
		and the durations of those artifacts.
		'''
		self.usability = self.n.general_notes['usability']
		self.nbc= len(self.bc.bad_channels)
		self.nbe= len(self.be.bad_epochs)
		self.clean_duration = sum([c.duration for c in self.bads if c.annotation =='clean'])
		self.artifact_duration = sum([c.duration for c in self.bads if c.annotation =='artifact'])
		self.clean_perc= self.clean_duration / self.block_duration
		self.artifact_perc= self.artifact_duration / self.block_duration

	def save(self):
		'''Write the object to xml.
		'''
		x = xml_handler.xml_handler(bads=self,artifact_type='bads')
		x.bads2xml()
		x.save(path.bads_annotations + 'bads_' + self.name + '.xml')


def read_notes(exp):
	'''Read all notes of all block of all participants OBSOLETE.
	'''
	names = read_names(exp)
	return [notes.note(name) for name in names]

def read_names(exp):
	'''Read all names of each block of all participants of one experiment.'''
	return [line for line in open(path.data+'names-sorted-duration_'+exp).read().split('\n') if line]


def get_xml_filenames(name):
	'''Get the filenames of the xml files for the corrected automatic 
	epoch and channel annotations.'''
	bad_epochs_filename = get_xml_filename(name,'bad_epoch')
	bad_channels_filename = get_xml_filename(name,'bad_channel')
	return bad_epochs_filename, bad_channels_filename

def get_xml_filename(name,bad_type = 'bad_epoch'):
	'''Get filename of the xml file for bad epoch or channel.'''
	if bad_type == 'bad_epoch': directory = path.corrected_artifact_cnn_xml
	elif bad_type == 'bad_channel': directory = path.corrected_ch_cnn_xml
	temp = glob.glob(directory + '*' + name + '*.xml')
	if temp != []: 
		if len(temp) > 1: print('found multiple files ERROR',temp)
		filename = temp[0]
		return filename
	else: print('could not find file',name)

def make_all_bads(exp,fo,force_make = True):
	'''Create bads for all blocks of a experiment.'''
	output = []
	names = read_names(exp)
	for i,name in enumerate(names):
		print(name,i,len(names))
		b = bads(name =name,fo = fo,force_make = force_make)
		print(b)
		output.append(b)
	return output

def load_all_bads(exp,fo,force_make = False):
	'''Load bads for all blocks of a experiment.'''
	output = []
	names = read_names(exp)
	for i,name in enumerate(names):
		print(name,i,len(names))
		b = bads(name =name,fo = fo,force_make = force_make)
		print(b)
		output.append(b)
	return output

	
def make_xml_name(name):
	'''Create filename for the bads annotation xml file.'''
	f = path.bads_annotations + 'bads_'+name+'.xml'
	return f
