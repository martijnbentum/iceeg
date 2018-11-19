import experiment as e
import glob
import manual_artifact_coder as mac
from  matplotlib import pyplot as plt
import load_all_ort
import notes
import os
import path
import utils
import windower


class corrector:
	'''Wrapper for manual artifact corrector.
	You can provide a names list ordered on number of automatically detected artifacts.
	This speeds up annotations (all clean blocks first)
	You can annotate epoch, channels and notes, with channels you can indicate
	to enforce only to show those blocks that are already epoch annotated.
	'''
	def __init__(self,coder_name,exp,bid = 1,pp_id = 1,fo = None,save_dir = '',annotation_type = 'corrector', skip_corrected = True,names_list = True):
		'''annotate all blocks of an experiment based on names list of blocks.
		coder_name 			name of annotator
		exp 				experiment name: o, k, ifdav
		bid 				block id (if no names list provided, start here)
		pp_id 				participant id (idem)
		fo 					fid2ort, speeds up loading participants; p = e.Participant, p.fid2ort
		save_dir 			provide other directory for saving
		annotation type 	corrector (epoch), channel_corrector (channel,note)
		skip_corrected 		do not show blocks that already have a file in save_dir
		names_list 			whether to use list of block file names
		'''
		
		self.coder_name = coder_name
		self.exp = exp
		self.bid = bid
		self.pp_id = pp_id
		self.annotation_type = annotation_type
		if fo == None: self.fo = load_all_ort.load_fid2ort()
		else: self.fo = fo
		if save_dir == '': 
			if annotation_type == 'corrector':
				self.save_dir = path.corrected_artifact_cnn_xml 
				if type(names_list) == list: self.names_list = names_list
				elif names_list == True: self.names_list = read_names(exp)
				self.coder = self.coder_name + '-' +self.annotation_type
			if annotation_type == 'channel_corrector': 
				self.save_dir = path.corrected_ch_cnn_xml
				if type(names_list) == list: self.names_list = names_list
				elif names_list == True: self.names_list = read_ch_names(exp)
				self.coder = self.coder_name + '-ch-corrector' 
		else: self.save_dir = save_dir
		self.annotation_type = annotation_type
		self.skip_corrected = skip_corrected
		self.done = False


	def next_bid(self):
		'''without names list, iterate through pp and bid.'''
		if self.exp == 'o' and self.bid == 7: self.done = True
		if self.exp == 'k' and self.bid == 21: self.done = True 
		if self.exp== 'ifadv' and self.bid == 6: self.done = True
		if self.done: print('done')
		else: self.bid += 1


	def next_pp_id(self):
		'''without names list, iterate through pp and bid.'''
		self.clean_up()
		if self.pp_id == 48: 
			self.pp_id = 1
			self.next_bid()
			if not self.done: self.load()
		else: 
			self.pp_id += 1
			try:self.load()
			except:
				print('Could not load pp: ',self.pp_id,self.exp,self.bid,'loading next pp')
				self.next_pp_id()


	def load(self):
		'''load a block for annotation, based on pp_id and bid (Sort of obsolete).'''
		self.p = e.Participant(self.pp_id,self.fo)
		if not hasattr(self,'s') or self.s.exp_type != 's'+self.exp:
			self.p.add_session(self.exp)
			self.s = getattr(self.p,'s'+self.exp)
		self.b = getattr(self.s,'b'+str(self.bid))
		self.name = windower.make_name(self.b)
		self.corrected_filename = self.coder + '_' + self.name + '.xml'
		if os.path.isfile(path.data +self.save_dir + self.corrected_filename) and self.skip_corrected:
			self.next_pp_id()
			return 0
		if self.annotation_type == 'corrector':
			self.m = mac.ac(self.b, coder = self.coder, annotation_type = self.annotation_type)
		elif self.annotation_type == 'channel_corrector':
			self.m = mac.ac(self.b, coder = self.coder, annotation_type = self.annotation_type,add_channel_to_remove = 'Fp2',enforce_coder=False)
		if len(self.m.bad_epochs) == 0:
			delattr(self,'m')
			self.next_pp_id()
			return 0
		
		
	def clean_up(self):
		'''Clean up the plot from the manual artifact annotator.'''
		plt.close('all')
		if hasattr(self,'m'): delattr(self,'m')
	
		
	def run(self):
		'''Iterate through pp_id/bid, sort of OBSOLETE.'''
		self.load()
		while 1:
			if not plt.get_fignums() and not self.done and self.m.stop_message != 'stop':
				self.next_pp_id()
			else: 
				print('last pp: ',self.pp_id,' bid: ',self.bid,' exp: ',self.exp)
				break

	def find_start(self):
		'''Find the first block to annotate, if skip_corrected is true.'''
		self.fn = glob.glob(self.save_dir + '*exp-'+self.exp+'*.xml')
		fn = self.fn
		self.corrected_names = ['pp' + f.split('_pp')[-1].split('.')[0] for f in fn]
		corrected_names = self.corrected_names
		print(self.names_list[:3],fn[:3],corrected_names[:3])
		for i,name in enumerate(self.names_list):
			if name not in corrected_names:
				print(name,'not yet corrected, starting here')
				return i
		return len(self.names_list)

	def load_next_name(self):
		'''Load next block for annotation (based on names_list).'''
		self.current_index += 1
		self.clean_up()
		self.name = self.names_list[self.current_index]
		self.corrected_filename = self.coder + '_' + self.name + '.xml'
		if os.path.isfile(self.save_dir + self.corrected_filename) and self.skip_corrected:
			self.load_next_name()
			return 0
		elif self.annotation_type == 'channel_corrector':
			n = path.corrected_artifact_cnn_xml + 'tim-corrector_' + self.name + '.xml'
			n1 = path.corrected_artifact_cnn_xml + 'martijn-corrector_' + self.name + '.xml'
			n2 = path.corrected_artifact_cnn_xml + 'rep-3_perc-50_fold-2_part-90_' + self.name + '.xml'
			if not os.path.isfile(n) and not os.path.isfile(n1) and not os.path.isfile(n2):
				print(n,'not found',self.name)
				print(n1,'not found')
				print(n2,'not found')
				self.load_next_name()
				return 0
		self.b = utils.name2block(name=self.name,fo= self.fo)
			
		# self.m = mac.ac(self.b, coder = self.coder, annotation_type = self.annotation_type)
		if self.annotation_type == 'corrector':
			self.m = mac.ac(self.b, coder = self.coder, annotation_type = self.annotation_type)
		elif self.annotation_type == 'channel_corrector':
			self.m = mac.ac(self.b, coder = self.coder, annotation_type = self.annotation_type,add_channel_to_remove = 'Fp2',enforce_coder=False)
		# self.m = mac.ac(self.b,save_dir = self.save_dir, coder = self.coder, annotation_type = self.annotation_type)
		if len(self.m.bad_epochs) == 0:
			self.load_next_name()
			return 0

	def run_names_list(self):
		'''Run through all block to be annotated based on names_list.'''
		print('running from names list, searching start index...')
		if self.skip_corrected: self.current_index = self.find_start()
		else: self.current_index = 0
		self.current_index -=1
		print('starting at index:',self.current_index)
		self.load_next_name()
		
		while 1:
			if not plt.get_fignums() and self.m.stop_message != 'stop':
				print('loading next name...')
				self.load_next_name()
			if self.current_index >= len(self.names_list) -1 or self.m.stop_message =='stop':
				print('last block:',self.name,'done with',self.exp)
				break


	def load_next_note(self):
		'''When annotating notes, this will check whether a note exists 
		and whether usability is annotated.
		'''
		self.clean_up()
		fn = glob.glob(path.notes + '*.xml')
		self.name = self.names_list[self.current_index]
		self.note = notes.note(self.name)
		if self.note.filename not in fn: annotate = True
		elif self.note.general_notes['usability'] == 'na': annotate = True
		else: annotate = False
		if annotate:
			self.b = utils.name2block(self.name,self.fo)
			self.m = mac.ac(self.b, coder = self.coder, annotation_type = self.annotation_type,add_channel_to_remove = 'Fp2',enforce_coder=False)
		else: 
			print(self.name,'already annotated')
			self.note.show()
		self.current_index +=1
		self.load_next_note()
		


	def run_notes(self):
		'''Annotate notes for all files that do not have a note.'''
		self.current_index = 0
		while 1:
			if not hasattr(self,'m'): 
				print('annotating first note')
				self.load_next_note()
			if not plt.get_fignums() and self.m.stop_message != 'stop':
				print('loading next note...')
				self.load_next_note()
			if self.current_index >= len(self.names_list) -1 or self.m.stop_message =='stop':
				print('last block:',self.name,'done with',self.exp)
				break
		

def read_names(exp):
	'''Get the names_list for a specific experiment, these are ordered on number of artifacts.'''
	return [line for line in open(path.data+'names-sorted-duration_'+exp).read().split('\n') if line]

def read_ch_names(exp):
	'''Get names_list for specific experiment for channel annotation, ordered on number of artifacts.'''
	return [line for line in open(path.data+'names-sorted-duration_channels_'+exp).read().split('\n') if line]
