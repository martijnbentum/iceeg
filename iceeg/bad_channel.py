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
			
	def fake_hash(self):
		return self.channel + '_' + str(self.epoch_id)

	def plot(self,channel_data,offset = None, color = None):
		'''plot start / end boundary object add transparent color over time window and plot annotation epoch id.'''
		if offset == None: offset = self.offset
		if not self.visible: return 0
		if self.ok and not self.plotted:
			if color != None:
				plt.plot(range(self.start.x,self.end.x,1), np.zeros(channel_data[self.start.x:self.end.x].shape)-75,alpha = 0.8, color = color, linestyle= self.linestyle,linewidth =10)
				plt.plot(range(self.start.x,self.end.x,1),np.zeros(channel_data[self.start.x:self.end.x].shape)+ 1075,alpha = 0.8, color = color, linestyle= self.linestyle,linewidth =10)
				plt.plot(range(self.start.x,self.end.x,1), channel_data[self.start.x:self.end.x] + offset,alpha = 0.8, color = color, linestyle= self.linestyle,linewidth =3)
				self.plot_annotation()
			else:
				if self.correct == 'correct':
					color = self.color
				else:
					color = 'grey'
				# plt.axvspan(self.start.x,self.end.x, facecolor = self.color, alpha = 0.1) 
				plt.plot(range(self.start.x,self.end.x,1), channel_data[self.start.x:self.end.x] + offset,color = 'white')
				plt.plot(range(self.start.x,self.end.x,1), channel_data[self.start.x:self.end.x] + offset,alpha = self.alpha, color = color, linestyle= self.linestyle,linewidth =6)

	def plot_annotation(self,plot_correct = True):
		'''Plot the annotation to the plot window.'''
		ylow,yhigh = plt.ylim()
		if self.ok:
			center = (self.st_sample + self.et_sample) / 2
			if self.correct == 'correct':
				plt.annotate('V', xy=(self.st_sample, ylow+30), color = 'green',fontsize=20)
				plt.annotate('V', xy=(self.et_sample - 50, ylow+30), color = 'green',fontsize=20)
			if self.correct == 'incorrect':
				plt.annotate('X', xy=(self.st_sample, ylow+30), color = 'red',fontsize=20)
				plt.annotate('X', xy=(self.et_sample - 50, ylow+30), color = 'red',fontsize=20)

	def set_complete_replot(self):
		'''Set flag to replot epoch (if something changed, e.g. boundary is deleted).'''
		self.plotted = False


	def in_plot_epoch(self,start,end):
		'''Check whether bad epoch and its boundaries are in the plot window defined by start and end.'''
		self.visible = False
		if self.ok:
			if self.end.x >= start and self.end.x <= end:
				self.visible = True
			if self.start.x < end and self.start.x > start:
				self.visible = True
			if self.start.x <= start and self.end.x >= end:
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


	def expand_to_zero(self, channel_data, seek_distance = 100):
		self.zero_crossing(channel_data,'start', seek_distance)
		self.zero_crossing(channel_data,'end', seek_distance)

	def zero_crossing(self,channel_data,boundary_type='start',seek_distance=100):
		d = channel_data
		# set boundary(start or end)
		if boundary_type == 'start': 
			boundary = self.start
			if boundary.x < seek_distance: start_index = 0
			else: start_index = boundary.x - seek_distance
			end_index = boundary.x
			seek = 'before'
		elif boundary_type == 'end':
			boundary = self.end
			cds = channel_data.shape[0]
			if  cds - boundary.x < seek_distance: end_index = cds-1
			else: end_index= boundary.x + seek_distance
			start_index = boundary.x
			seek = 'after'
		else: raise ValueError('unknown boundary type',boundary_type)
		if channel_data[boundary.x] < 1 and channel_data[boundary.x] > -1: 
			# boundary is already close to zero, do nothing
			print(boundary,channel_data[boundary.x],boundary.x,'do nothing')
			return
		thres = 3
		while 1:
			# find indices where channel is close to 0
			zero_indices = np.where((d < thres) & (d > -1*thres))[0]
			cz, cd = find_closest_zero(zero_indices,boundary.x,seek)
			closest_zero, closest_diff = cz, cd
			if closest_diff < seek_distance:
				new_x = closest_zero
				print('found closest zeros:',closest_zero,closest_diff)
				break
			else: thres *= 1.5
		self.set_boundary(boundary,new_x)


	def set_boundary(self,boundary, new_x):
		boundary.old_boundary = boundary.x
		boundary.x = new_x
		self.set_info()
		self.set_complete_replot()

	def expand_to_bad_channel(self,other):
		if other == None and type(other) != bad_epoch.Boundary: 
			print('cannot expand to:',other==None,type(other)!=bad_epoch.Boundary)
			return
		if other.end.x < self.start.x: self._expand_to_before(other)
		elif self.start.x < other.end.x: self._expand_to_next(other)

	def _expand_to_before(self,other):
		if other.end.x >= self.start.x: return 
		else: self.set_boundary(self.start, other.end.x)

	def _expand_to_next(self,other):
		if self.end.x >= other.start.x: return 
		else: self.set_boundary(self.end, other.start.x)
			
	def reset_old_boundaries(self):
		start, end = self.start, self.end
		if hasattr(start,'old_boundary'):
			 start.x,start.old_boundary = start.old_boundary,start.x
		if hasattr(end,'old_boundary'):
			 end.x,end.old_boundary = end.old_boundary,end.x
		self.set_info()
		self.set_complete_replot()


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



def find_closest_zero(zero_indices,x,seek):
	closest_zero = 'Na'
	zero_distance = 10**8
	closest_diff = 10**8
	for zeroi in zero_indices:
		diff = x - zeroi
		if seek == 'before' and diff > 0 and diff < zero_distance: 
			closest_zero = zeroi
			zero_distance= diff
		elif seek == 'after' and diff < 0 and abs(diff) < zero_distance: 
			closest_zero = zeroi
			zero_distance= abs(diff)
	return closest_zero, zero_distance


