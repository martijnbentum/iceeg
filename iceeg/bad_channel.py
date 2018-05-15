import bad_epoch
import copy
import numpy as np
from matplotlib import pyplot as plt


class Bad_channel(bad_epoch.Bad_epoch):
	'''Object to store information about streches of EEG channel data that have artifacts.
	Contains boundaries (object) of type start and end with start sample with respect to start block
	Contains info about participant id experiment type block number and start sample of block
	'''
	def __init__(self,channel = '', epoch_id =None ,epoch_ids= None,start_boundary = None,end_boundary = None,annotation = 'garbage',coder = 'martijn',color = '', pp_id = None, exp_type = None, block_st_sample = None, bid = None, visible = True, correct = 'correct',block_et_sample = None,alpha = .2,linestyle= '-',offset = 0, perc_clean =-9):
		'''Information about stretch of eeg data containing artefacts.
		channel 			the channel name

		start_boundary 		boundary object with start sample number of type start
		end_boundary 		boundary object with end sample number of type end
		annotation 			label for the type of artefact
		color 				color of epoch in plot
		...
		'''
		super().__init__(start_boundary=start_boundary,end_boundary=end_boundary,annotation=annotation,coder=coder,pp_id=pp_id,exp_type=exp_type,block_st_sample=block_st_sample,bid=bid,visible=visible,correct=correct,block_et_sample=block_et_sample,epoch_id = epoch_id,epoch_ids = epoch_ids,perc_clean=perc_clean)
		self.channel = channel
		self.alpha = alpha
		self.linestyle= linestyle
		self.offset = 0
		self.color_dict = {'heog':'green','jump':'cyan','garbage':'black','unk':'red','maybe':'purple','all':'grey','high-frequency':'gold','clean':'white','artifact':'blue'}
		if color == '':
			self.color = self.color_dict[annotation] 


	def __str__(self):
		m = 'bad_channel id:\t\t'+str(self.epoch_id) + '\n'
		if self.ok:
			m += 'channel:\t\t\t'+str(self.channel) +'\n'
			m += 'pp id:\t\t\t'+str(self.pp_id) +'\n'
			m += 'exp type:\t\t'+str(self.exp_type)+'\n'
			m += 'block number:\t\t'+str(self.bid)+'\n'
			m += 'block_st_sample:\t'+str(self.block_st_sample)+'\n'
			m += 'st_sample:\t\t'+str(int(self.st_sample))+ '\n'
			m += 'et_sample:\t\t'+str(int(self.et_sample))+ '\n'
			m += 'duration:\t\t'+str(int(self.duration)) + '\n'
			m += 'annotation:\t\t'+str(self.annotation) + '\n'
		m += 'visible:\t\t'+str(self.visible)+ '\n'
		m += 'ok:\t\t\t'+str(self.ok)+ '\n'
		return m

	def __repr__(self):
		if self.ok: st, et = self.st_sample, self.et_sample
		else: st, et = 'NA','NA'
		return 'Bad_channel-object:\t'+ self.channel+'\t' + self.annotation.ljust(9)+ '\tstart: '+str(st) + '\tend: '+str(et) + '\tcoder: ' +self.coder + '\tok: ' + str(self.ok)
			

	def __eq__(self,other):
		if type(other) != Bad_channel: return False
		return self.start == other.start and self.end == other.end and self.channel == other.channel

	def __lt__(self,other):
		if type(other) != Bad_channel: return None
		if hasattr(self,'start') and hasattr(other, 'start'):
			return self.start < other.start

	def __gt__(self,other):
		if type(other) != Bad_channel: return None
		if hasattr(self,'start') and hasattr(other, 'start'):
			return self.start > other.start
			

	def plot(self,channel_data,offset = None):
		'''plot start / end boundary object add transparent color over time window and plot annotation epoch id.'''
		if offset == None: offset = self.offset
		if not self.visible: return 0
		if self.ok and not self.plotted:
				# plt.axvspan(self.start.x,self.end.x, facecolor = self.color, alpha = 0.1) 
				plt.plot(range(self.start.x,self.end.x,1), channel_data[self.start.x:self.end.x] + offset,color = 'white')
				plt.plot(range(self.start.x,self.end.x,1), channel_data[self.start.x:self.end.x] + offset,alpha = self.alpha, color = self.color, linestyle= self.linestyle,linewidth =6)


	def set_complete_replot(self):
		'''Set flag to replot epoch (if something changed, e.g. boundary is deleted).'''
		self.plotted = False


	def in_plot_epoch(self,start,end):
		'''Check whether bad epoch and its boundaries are in the plot window defined by start and end.'''
		self.visible = False
		if self.ok:
			if self.end.x > start and self.end.x < end:
				self.visible = True
			if self.start.x < end and self.start.x > start:
				self.visible = True
			if self.start.x < start and self.end.x > end:
				self.visible = True
			self.start.in_plot_epoch(start,end)
			self.end.in_plot_epoch(start,end)


	def in_bad_channel(self,x):
		'''Check whether x is within the bad epoch (to check whether mouse position is in this epoch).'''
		if self.ok and x >= self.start.x and x <= self.end.x: return True
		else: return False
		

	def check_boundaries(self,channel_data = None):
		'''Check the state start and end boundaries, whether they are both there.'''
		self.start_boundary_ok = True 
		self.end_boundary_ok = True 
		self.last_ok = copy.copy(self.ok)
		self.ok = True
		self.empty = False
		if self.start == None:
			self.start_boundary_ok = False
		if self.end == None:
			self.end_boundary_ok = False

		if not self.start_boundary_ok or not self.end_boundary_ok:
			self.ok = False
		if not self.start_boundary_ok and not self.end_boundary_ok:
			print('epoch is empty')
			self.empty = True
		self.set_info()
		if self.ok and type(channel_data) == np.ndarray: self.plot(channel_data)

		

	def del_boundary(self,boundary):
		'''Delete the boundary that equals boundary or delete the boundary specified by the string 'start' or 'end'.'''
		self.redraw = True
		if boundary == 'start' or self.start == boundary:  
			self.start = None
		elif boundary == 'end' or self.end == boundary: 
			self.end = None
		else:
			self.redraw = False
			print('boundary not present in epoch')
		self.check_boundaries()


	def get_sample_info(self,multiplier =1):
		self.check_boundaries()
		if self.ok:
			return self.start.x * multiplier, self.end.x * multiplier, self.end.x * multiplier - self.start.x * multiplier
		else: return None
