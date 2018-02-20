import bad_epoch
import copy
from matplotlib import pyplot as plt
import numpy as np
import os
import path
from scipy import stats
import string
import time
import utils
import windower
import xml_handler

class ac:
	def __init__(self,b,length = 10,overlap = False,coder = 'martijn',filename = '',load_xml = True,sf = 100, remove_ch= None):
		'''Interface to easily annotate eeg signal
		b 			block object
		length 		duration in seconds of an epoch in the interface
		decimate 	take sample every n samples, speeds up plotting
		overlap 	whether plot windows should overlap
		coder 		Name of the coder (if automatic it should be computer)
		filename 	specify xml file that is loaded for bad_epochs, default is to generate filename based on block info
		load_data 	surpress previously generated bad_epochs (create new annotation), old versions are moved to OLD 
					directory in artifacts folder.
		'''
		self.artifact_index = -1 
		self.old_epoch_id = ''
		self.key_dict = {'h':'heog','d':'drift','a':'alpha','g':'garbage','m':'movement','u':'unk','x':'incorrect','V':'correct','j':'jump','c':'ch-jump'}
		self.redraw = False
		self.b = b
		if remove_ch != None and type(remove_ch) == list: self.remove_ch = remove_ch
		else: self.remove_ch = ['VEOG','HEOG','TP10_RM','STI 014','LM']
		self.load_eeg()
		self.set_info()
		self.filename = filename
		self.length = int(float(length) * sf)
		self.e_index = 0
		self.event_dict = {}
		self.overlap = overlap
		self.coder = coder
		self.boundaries = []
		self.bad_epochs = []
		self.last_bad_epoch_id = int(open(path.artifacts +'last_bad_epoch_id').read())
		self.load_from_xml(filename)
		self.last_save = time.time()
		self.make_epoch()
		self.plot_epoch('all')
		self.reset_visible()
		self.handle_plot(True)
		self.redraw = False
		self.run()

	def load_eeg(self):
		self.ch_names = utils.load_ch_names()
		self.block_name = windower.make_name(self.b)
		self.data = utils.load_100hz_numpy_block(self.block_name) * 10**6
		self.remove_channels(self.remove_ch)

	def remove_channels(self,channels = []):
		self.remove_ch += channels
		self.ch_mask = [n not in self.remove_ch for n in self.ch_names]
		self.ch_names= [n for n in self.ch_names if not n in self.remove_ch]
		self.data = self.data[self.ch_mask,:]


	def load_from_xml(self,filename = ''):
		if os.path.isfile(filename): self.filename = filename
		elif os.path.isfile(path.artifacts + filename): self.filename = path.artifacts + filename
		elif os.path.isfile(self.filename): pass
		else:
			print('Auto generating filename based on block information.')
			self.filename = path.artifacts + self.coder + '_pp' + str(self.pp_id) + '_exp-' + self.exp_type + '_bid-' + str(self.bid) + '.xml'
		if os.path.isfile(self.filename): 
			xml = xml_handler.xml_handler(filename = self.filename)
			self.bad_epochs = xml.xml2bad_epochs(multiplier = 0.1)
			for be in self.bad_epochs:
				if be.start == None or be.end == None:
					print(be)
				else:
					self.boundaries.append(be.start)
					self.boundaries.append(be.end)

	def handle_save_xml(self,force_save = False):
		save_ok = False
		for be in self.bad_epochs:
			if be.ok: save_ok = True
		if save_ok and (time.time() - self.last_save > 60 or force_save == True):
			print('saving:',self.filename)
			print('nbad epochs:',len(self.bad_epochs))
			self.last_save = time.time()
			xml = xml_handler.xml_handler(self.bad_epochs,self.filename)
			xml.bad_epochs2xml(multiplier = 10)
			xml.write()
			

	def set_info(self):
		'''Set experimental info (participant id, experiment type, etc.) to current object.'''
		self.exp_dict = {'ifadv':1,'o':2,'k':3,1:'ifadv',2:'o',3:'k'}
		b = self.b
		self.pp_id, self.exp_type, self.bid,self.block_st_sample = b.pp_id, b.exp_type, b.bid, b.st_sample
		self.exp_id = '9' 


	def make_epoch(self):
		'''Create a start and end numpy array with start and end times of plot windows.
		Length (specified in seconds set the length of the window, overlap specifies whether the plot
		windows should overlap half of their length.'''
		if self.overlap:
			# If overlap is true make it overlap for half of the window
			self.start_epoch= np.arange(0,self.data.shape[1],int(self.length/2))
		else:
			self.start_epoch= np.arange(0,self.data.shape[1],self.length)
		self.end_epoch= self.start_epoch+ self.length
		# last epoch can only last until end data
		self.end_epoch[-1] = self.data.shape[1] 


	def run(self):
		'''Ask user input, used as an hack to catch input intended for the pyplot.'''
		a = input('b/n:')
		print('you provided:',a)
		self.handle_epoch_switch(a)
		if a == 'exit' or a == 'q': 
			self.running = False
			return 0
		self.run()


	def handle_epoch_switch(self,a):
		'Switch to next plot epoch.'''
		if a == 'n':
			self.e_index += 1
			if self.e_index == len(self.start_epoch): self.e_index = 0
		if a == 'b':
			self.e_index -= 1
			if self.e_index < 0: self.e_index = len(self.start_epoch) - 1
		if a == 'n' or a == 'b':
			self.reset_visible()
			self.redraw_plot()


	def handle_epoch_jump(self,n):
		'''Set plot window to start (1) or end (0) or proportional to number 2 - 9.
		if there are 100 plot epochs, 1 sets first plot epoch and 5 sets 50th plot epoch'''
		if n == '0': i = len(self.start_epoch) - 1
		elif n == '1': i = 0
		else: i = int(len(self.start_epoch) / 10) * int(n)
			
		if not self.e_index == i:
			self.e_index = i
			self.reset_visible()
			self.redraw_plot()


	def reset_visible(self):
		'''Check whether each bad_epoch is visible in the current plot epoch and set flag in the be accordingly.'''
		for be in self.bad_epochs:
			be.in_plot_epoch(self.start_epoch[self.e_index],self.end_epoch[self.e_index])


	def handle_plot(self,force_redraw = False):
		'''Check whether something has changed that requires a redraw of the plot window. Peform redraw when necessary.'''
		self.fig.canvas.draw()
		self.check_redraw()
		if self.redraw or force_redraw:
			self.redraw_plot()
			self.redraw = False


	def check_redraw(self):
		'''Check whether something has changed that requires a redraw of the plot window.'''
		for be in self.bad_epochs:
			if be.visible and be.redraw:
				self.redraw = True
				be.redraw = False
				

	def redraw_plot(self):
		'''Redraw current plot window (because something has been removed.)'''
		plt.close(self.fig)
		for be in self.bad_epochs:
			be.set_complete_replot()
		self.plot_epoch('all')


	def find_before_and_after_boundaries(self):
		'''aggregate all before and all after boundaries (relative to mouse) 
		in seperate lists of ascending order (distance from mouse).'''
		self.boundaries.sort()
		self.before_boundaries, self.after_boundaries = [] , []
		for i,b in enumerate(self.boundaries):
			if b.x < self.event.xdata:
				self.before_boundaries.append(b)
			if b.x > self.event.xdata:
				self.after_boundaries.append(b)
		self.before_boundaries.reverse()
			

	def get_bad_epoch(self,boundary):
		'''return bad epoch that contains this boundary.
		boundaries should alway be contained in a bad epoch.'''
		for be in self.bad_epochs:
			if boundary in be:
				return be


	def find_completion_bad_epoch(self,boundary_type):
		'''find bad epoch that is closest in time with missing boundary of correct type within 2 plot epochs of mouse.
		boundary_type 		start or end, specifying boundary type that completes current boundary i.e. if a start 
							boundary is made an epoch should be searched with only an end boundary'''
		if boundary_type =='end': boundaries = self.after_boundaries
		elif boundary_type == 'start': boundaries = self.before_boundaries
		else: return 0
		for b in boundaries:
			if abs(b.x - self.event.xdata) > self.length * 2 * 1000:
				return False
			if b.boundary_type == boundary_type:
				be = self.get_bad_epoch(b)
				if be == None: return False # getting errors be are none
				if boundary_type== 'start' and be.end == None: return be
				if boundary_type== 'end' and be.start== None: return be
		return False
		

	def delete_bad_epoch(self,epoch_id):
		'''Delete a bad_epoch object based on epoch_id.'''
		index = -1
		print('deleting epoch')
		print([epoch_id])
		print('n bad epochs',len(self.bad_epochs))
		for i,be in enumerate(self.bad_epochs):
			if be.epoch_id == epoch_id:
				index = i
				break
		print(i)
		if index > -1:
			be = self.bad_epochs.pop(index)
			print('removing following epoch')
			print(be)
			print('n bad epochs',len(self.bad_epochs))
			

	def add_zeros(self,goal_length,number):
		l = len(str(number))
		nzeros = goal_length - l
		if nzeros > 0: return '0' * nzeros + str(number)
		else: return str(number)
			

	def make_bad_epoch_id(self):
		'''Create an unique integer id.'''
		self.last_bad_epoch_id += 1
		fout = open(path.artifacts +'last_bad_epoch_id','w')
		fout.write(str(self.last_bad_epoch_id))
		fout.close()
		return self.last_bad_epoch_id


	def annotate_bad_epoch(self,annotation = ''):
		'''Set the label for the bad epoch.'''
		boundaries = []
		for be in self.bad_epochs:
			if be.ok and be.visible and be.in_bad_epoch(self.mousex):
				boundaries.extend([be.start,be.end])
		dist = self.length * 1000 * 2 
		if len(boundaries) == 0: return 0
		for b in boundaries:
			if abs(b.x - self.mousex) < dist:
				closest = b
		be = self.get_bad_epoch(closest)	
		print(self.mousex)
		print(be)
		if annotation == 'correct' or annotation == 'incorrect':
			be.set_correct(annotation)
		else: be.set_annotation(annotation)


	def handle_start(self):
		'''Create a start boundary, and either add this to closest end boundary or create new epoch.'''
		boundary = bad_epoch.Boundary(self.event.xdata,'start')
		self.boundaries.append(boundary)
		be = self.find_completion_bad_epoch(boundary_type = 'end')
		if be:
			print('combining boundaries')
			be.set_start(boundary)
		else:
			print('making new epoch')
			self.bad_epochs.append(bad_epoch.Bad_epoch(start_boundary = boundary, pp_id = self.pp_id, coder = self.coder,exp_type = self.exp_type, bid = self.bid, block_st_sample = self.block_st_sample, epoch_id = self.make_bad_epoch_id()))
			

	def handle_end(self):
		'''Create a end boundary, and either add this to closest start boundary or create new epoch.'''
		boundary = bad_epoch.Boundary(self.event.xdata,'end')
		self.boundaries.append(boundary)
		be = self.find_completion_bad_epoch(boundary_type = 'start')
		if be:
			print('combining boundaries')
			be.set_end(boundary)
		else:
			self.bad_epochs.append(bad_epoch.Bad_epoch(end_boundary = boundary, pp_id = self.pp_id, exp_type = self.exp_type, bid = self.bid, block_st_sample = self.block_st_sample, epoch_id = self.make_bad_epoch_id()))


	def handle_delete(self):
		'''Delete boundary that is closest to the mouse cursor but not further than 30 away.'''
		before, after = 100, 100
		if len(self.before_boundaries) > 0: before = abs(self.before_boundaries[0].x - self.mousex)
		if len(self.after_boundaries) > 0: after =  abs(self.after_boundaries[0].x - self.mousex)
		if before > 30 and after > 30: return 0
		if before < after: boundary = self.before_boundaries[0]
		if before > after: boundary = self.after_boundaries[0]
		be = self.get_bad_epoch(boundary)
		self.boundaries.pop(self.boundaries.index(boundary))
		be.del_boundary(boundary)
		if be.empty: 
			self.delete_bad_epoch(be.epoch_id)
			self.redraw = True


	def handle_bad_channel(self):
		'''Create a bad channel object. Work In Progress'''
		closest = 2000
		if self.event.xdata < self.start_epoch[self.e_index]: ch_y = self.ch_starty
		else: ch_y = self.ch_endy
		for i,y in enumerate(ch_y):
			delta = abs(self.event.ydata - y)
			if delta < closest: 
				closest = delta
				ch_index = i


	def find_next_artifact_epoch(self):
		artifact_names = ['garbage','unk','drift']
		epoch_index = self.e_index + 1
		while True:
			for be in self.bad_epochs:
				if be.annotation in artifact_names:
					if self.start_epoch[epoch_index] <= be.start.x <= self.end_epoch[epoch_index]:
						if be.epoch_id != self.old_epoch_id:
							self.old_epoch_id = be.epoch_id
							return epoch_index
			epoch_index += 1
			if epoch_index >= len(self.start_epoch):
				print('full circle LAST BAD EPOCH LAST BAD EPOCH\n'*30)
				return len(self.start_epoch) - 1


	def next_artifact_index(self):
		artifact_names = ['garbage','unk','drift']
		self.artifact_indices = [i for i,be in enumerate(self.bad_epochs) if be.annotation in artifact_names]
		current_index = self.artifact_index
		while True:
			self.artifact_index += 1 
			if self.artifact_index >= len(self.bad_epochs): 
				self.artifact_index = 0
				print('full circle LAST BAD EPOCH LAST BAD EPOCH\n'*30)
				break
			be = self.bad_epochs[self.artifact_index]
			if be.annotation in artifact_names:
				break

	def jump_to_next_artifact(self):
		new_e_index = self.find_next_artifact_epoch()
		if self.e_index != new_e_index: 
			self.e_index = new_e_index
			self.reset_visible()
			self.handle_plot(force_redraw = True)
		else: self.jump_to_next_artifact()
		
			

	def jump_to_previous_artifact(self):
		self.artifact_index -= 1 
		print('artifact_index:',self.artifact_index, 'going backwards')
		if self.artifact_index < 0: self.artifact_index = len(self.bad_epochs) 
		be = self.bad_epochs[self.artifact_index]
		for index in range(len(self.start_epoch)):
			if self.start_epoch[index] <= be.start.x <= self.end_epoch[index]:
				self.e_index = index
				break
		self.reset_visible()
		self.handle_plot(force_redraw = True)
			

		


	def on_click(self,event):
		'''Handle click event - links to pyplot window event manager.'''
		self.event = event
		if self.event.xdata != None and self.event.ydata != None:
			print(self.event.xdata, self.event.ydata)
			# self.find_closest_epoch()
			self.find_before_and_after_boundaries()
			# if self.event.xdata < self.start_epoch[self.e_index]  or self.event.xdata > self.end_epoch[self.e_index]:
				# self.bad_channel()
			if self.event.button == 1: # left mouse button
				self.handle_start()
			elif self.event.button == 2: # scroll wheel
				self.handle_delete()
			elif self.event.button == 3: # right mouse button
				self.handle_end()
		self.last_event = event
		self.handle_save_xml()
		self.handle_plot()


	def on_key(self,event):
		'''Handle a key press event - links to pyplot window event manager.'''
		self.event_key = event
		print(event.key)
		force_save = False
		if event.key in ['b','n']:
			self.handle_epoch_switch(event.key)
		if event.key == '`':
			self.jump_to_next_artifact()
		# if event.key == 'z':
			# self.jump_to_next_artifact()
		if event.key in self.key_dict.keys():
			self.annotate_bad_epoch(self.key_dict[event.key])
		if event.key == ' ': force_save = True
		if event.key in string.digits:
			self.handle_epoch_jump(event.key)
		if event.key in ['p','z']: self.purge_bad_epochs()
		self.handle_plot()
		self.handle_save_xml(force_save)
		self.last_event_key = event

	def purge_bad_epochs(self):
		for be in self.bad_epochs:
			if not be.ok:
				if be.start: self.boundaries.pop(self.boundaries.index(be.start))
				elif be.end: self.boundaries.pop(self.boundaries.index(be.end))
				self.delete_bad_epoch(be.epoch_id)
		self.find_before_and_after_boundaries()
				

	def on_motion(self,event):
		'''Update mouse position.'''
		self.mousex = event.xdata
		self.mousey = event.ydata


				

	def plot_epoch(self,channels = ['Fz','Cz','Pz'],offset_value = 40,show_bad_epoch = True):
		'''Plot a window with specified channels of eeg data.
		channels 		names or indices of eeg channels, can also be all to plot all channels present
		offset_value 	vertical distance between eeg channels
		show_bad_epoch 	whether to show the bad epochs for this time window
		'''
		if channels == []: channels = 'all'
		if channels == 'all': channels = self.ch_names
		if type(channels[0]) == int: 
			self.channel_index= copy.deepcopy(channels)
			self.channels = [self.ch_names[i] for i in channels]
		else: 
			self.channel_index = [self.ch_names.index(n) for n in channels]
			self.channels = channels

		print(len(self.channel_index),len(self.channels),self.data.shape,self.end_epoch[-1])
		#Create figure
		# self.fig, self.ax = plt.subplots(num=None, figsize=(21, 11), dpi=80 )
		self.fig, self.ax = plt.subplots(num=None, figsize=(20, 9), dpi=80 )
		self.fig.tight_layout()
		self.fig.patch.set_facecolor('ivory')
		self.ax.set_facecolor('ivory')
		self.fig.canvas.set_window_title('pp'+ str(self.pp_id) + '  ' + self.exp_type + ' b' + str(self.bid) + '    nbe' + str(len(self.bad_epochs)))

		#Create handles that connect to the pyplot for mouse clicks, key presses and mouse motion.
		self.cidm = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.cidk = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


		# data is lowpass filtered at 30 Hz, nyquist = 60 Hz and sf = 1000, so
		start_index,end_index = self.start_epoch[self.e_index], self.end_epoch[self.e_index]
		yx = np.arange(start_index,end_index,1)
		y = self.data[:,yx]
			
		offset = 0
		self.ch_starty = np.zeros(len(self.channel_index))
		self.ch_endy = np.zeros(len(self.channel_index))

		clist = 'darkblue,brown,c,m,g,crimson'.split(',')
		ci =0
		for i in self.channel_index: 
			# plot a channel
			plt.plot(yx,y[i,:]+offset,linewidth = 0.9,color=clist[ci])
			ci +=1
			if ci == len(clist):ci =0
			# create coordinates for channel names before and after channel plot
			self.ch_starty[i] = y[i,0] + offset - 10 # subtract 10 to center name in y dimension
			self.ch_endy[i] = y[i,-1] + offset - 10 # subtract 10 to center name in y dimension
			# offset between channels
			offset += offset_value 

		#Set plot dimensions
		plt.ylim((-80,(len(self.channel_index) + 2) * 40))
		plt.xlim(start_index -50,start_index + self.length+50)


		#Plot channel name for each channel in the same color
		ci = 0
		for i,y in enumerate(self.ch_starty):
			plt.annotate(self.channels[i],xy=(start_index-30,y),color = clist[ci],fontsize=18)
			ci += 1
			if ci == len(clist): ci = 0 
		ci = 0
		for i,y in enumerate(self.ch_endy):
			plt.annotate(self.channels[i],xy=(end_index + 20,y),color=clist[ci],fontsize=18)
			ci += 1
			if ci == len(clist): ci = 0 

		#plot bad epochs
		if show_bad_epoch:
			for be in self.bad_epochs:
				be.plot()
		plt.grid(alpha=0.2,color='tan',lw=2)
		if self.e_index == len(self.start_epoch) - 1:
			plt.annotate('END',xy=(self.end_epoch[-1]-400, 1040), color ='red',fontsize=70)
		elif self.e_index == 0:
			plt.annotate('START',xy=(self.end_epoch[0]-800, 1065), color ='blue',fontsize=50)
			



'''
	def find_event(self,event):

		for k in self.event_dict.keys():
			if abs(k - event.xdata) < 15:
				return k
		return None

'''
