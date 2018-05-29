import experiment as e
import manual_artifact_coder as mac
from  matplotlib import pyplot as plt
import load_all_ort
import os
import path
import windower


class corrector:
	def __init__(self,coder_name,exp,bid = 1,pp_id = 1,fo = None,save_dir = '',annotation_type = 'corrector', skip_corrected = True):
		self.coder_name = coder_name
		self.exp = exp
		self.bid = bid
		self.pp_id = pp_id
		if fo == None: self.fo = load_all_ort.load_fid2ort()
		else: self.fo = fo
		if save_dir == '': self.save_dir = 'CORRECTED_ARTIFACT_CNN_XML/'
		else: self.save_dir = save_dir
		self.annotation_type = annotation_type
		self.skip_corrected = skip_corrected
		self.coder = self.coder_name + '-' +self.annotation_type
		self.done = False


	def next_bid(self):
		if self.exp == 'o' and self.bid == 7: self.done = True
		if self.exp == 'k' and self.bid == 21: self.done = True 
		if self.exp== 'ifadv' and self.bid == 6: self.done = True
		if self.done: print('done')
		else: self.bid += 1


	def next_pp_id(self):
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
		self.m = mac.ac(self.b,save_dir = self.save_dir, coder = self.coder, annotation_type = self.annotation_type)
		if len(self.m.bad_epochs) == 0:
			delattr(self,'m')
			self.next_pp_id()
			return 0
		
		
	def clean_up(self):
		plt.close('all')
		if hasattr(self,'m'): delattr(self,'m')
	
		
	def run(self):
		self.load()
		while 1:
			if not plt.get_fignums() and not self.done and self.m.stop_message != 'stop':
				self.next_pp_id()
			else: 
				print('last pp: ',self.pp_id,' bid: ',self.bid,' exp: ',self.exp)
				break
