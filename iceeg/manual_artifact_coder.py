import bad_channel
import bad_epoch
import copy
import glob
from matplotlib import pyplot as plt
import notes
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
	def __init__(self,b,length = 10,overlap = False,coder = 'martijn',filename = '',load_xml = True,sf = 100, remove_ch= None, show_cnn_pred = False, offset_value = 40, view_mode = 'all',filename_channels = '',default_annotation = 'garbage',default_annotation_channel = 'garbage', annotation_type = 'artifact',save_dir = ''):
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
		self.show_complete_bad = True
		self.artifact_selected = None
		self.b = b
		self.coder = coder
		self.annotation_type = annotation_type
		self.save_dir = save_dir
		if self.save_dir != '' and self.save_dir[-1] != '/': self.save_dir += '/'
		if annotation_type != 'artifact' and save_dir == '': raise ValueError('provide save dir to save corrections.')
		self.filename = filename
		self.filename_channels = filename_channels
		if annotation_type == 'corrector' and filename == '': self.find_cnn_xml_filename()
		self.default_annotation = default_annotation
		self.default_annotation_channel = default_annotation_channel
		self.channel_mode = 'off'
		self.channel_mode_index = 0
		self.heog_view= 'off'
		self.view_mode = view_mode
		self.sf = sf
		self.artifact_index = -1 
		self.old_epoch_id = ''
		self.key_dict = {'h':'heog','d':'drift','a':'alpha','g':'garbage','u':'unk','x':'incorrect','v':'correct','j':'jump','c':'ch-jump'}
		self.key_dict_channel = {'h':'heog','g':'garbage','u':'unk','x':'incorrect','v':'correct','j':'jump','m':'maybe','f':'high-frequency'}
		self.redraw = False
		self.show_cnn_pred = show_cnn_pred
		if remove_ch != None and type(remove_ch) == list: self.remove_ch = remove_ch
		else: self.remove_ch = ['VEOG','HEOG','TP10_RM','STI 014','LM']
		self.load_eeg()
		self.set_info()
		self.length = int(float(length) * sf)
		self.e_index = 0
		self.event_dict = {}
		self.overlap = overlap
		self.boundaries = []
		self.bad_epochs = []
		self.channel_boundaries = []
		self.bad_channels = []
		self.complete_bad_channel = []
		self.last_bad_epoch_id = int(open(path.artifacts +'last_bad_epoch_id').read())
		self.load_from_xml(filename)
		self.last_save = time.time()
		self.make_epoch()
		self.offset_value = offset_value
		self.plot_epoch('all',offset_value = self.offset_value)
		self.reset_visible()
		self.handle_plot(True)
		self.redraw = False
		self.run()

	def find_cnn_xml_filename(self):
		filename = path.data + self.save_dir + self.coder + '_' + windower.make_name(self.b) + '.xml'
		if os.path.isfile(filename): 
			self.filename = filename
			return 0
		name = windower.make_name(self.b) + '.xml'
		fn = glob.glob(path.artifact_cnn_xml + '*' +name)
		if len(fn) != 1: raise ValueError('did not find a unique filename',fn,name)
		self.filename = fn[0]

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
		elif self.show_cnn_pred:
			name = windower.make_name(self.b)
			fn = glob.glob(path.artifact_cnn_xml + '*' + name + '*')
			if len(fn) > 0: self.filename = fn[0]
		else:
			print('Auto generating filename based on block information.')
			self.filename = path.artifacts + self.coder + '_pp' + str(self.pp_id) + '_exp-' + self.exp_type + '_bid-' + str(self.bid) + '.xml'
			self.filename_channels = path.bad_channels+ self.coder + '_pp' + str(self.pp_id) + '_exp-' + self.exp_type + '_bid-' + str(self.bid) + '_channels.xml'

		if os.path.isfile(self.filename): 
			xml = xml_handler.xml_handler(filename = self.filename)
			self.bad_epochs = xml.xml2bad_epochs(multiplier = 0.1,remove_clean = True)
			for be in self.bad_epochs:
				if be.start == None or be.end == None:
					print(be)
				else:
					self.boundaries.append(be.start)
					self.boundaries.append(be.end)

		if self.filename != '' and self.filename_channels == '':
			self.filename_channels = self.filename.strip('.xml') + '_channels.xml'
			if not os.path.isfile(self.filename_channels): 
				name= self.filename_channels.split('/')[-1]
				self.filename_channels = path.bad_channels + name
			
		print(self.filename_channels)

		if os.path.isfile(self.filename_channels): 
			print('loading xml channels')
			xml = xml_handler.xml_handler(filename = self.filename_channels)
			self.bad_channels= xml.xml2bad_channels(multiplier = 0.1)
			for bc in self.bad_channels:
				if bc.annotation == 'all':self.complete_bad_channel.append(bc.channel)
				if bc.start == None or bc.end == None:
					print(bc)
				else:
					self.channel_boundaries.append(bc.start)
					self.channel_boundaries.append(bc.end)
		else: print('filename xml channels not found.',self.filename_channels)


	def handle_save_xml(self,force_save = False):
		if self.annotation_type == 'corrector':
			filename = path.data + self.save_dir + self.coder + '_' + windower.make_name(self.b) + '.xml'
		elif self.save_dir != '':
			filename = path.data + self.save_dir + self.filename.split('/')[-1]
		else: filename = self.filename
		save_ok = False
		for be in self.bad_epochs:
			if be.ok: save_ok = True
		if save_ok and (time.time() - self.last_save > 60 or force_save == True):
			print('saving:',self.filename)
			print('nbad epochs:',len(self.bad_epochs))
			self.last_save = time.time()
			xml = xml_handler.xml_handler(bad_epochs=self.bad_epochs,filename=filename)
			xml.bad_epochs2xml(multiplier = 10)
			xml.write()

		save_ok = False
		for be in self.bad_channels:
			if be.ok: save_ok = True
		if save_ok and (time.time() - self.last_save > 60 or force_save == True):
			print('saving:',self.filename)
			print('nbad channels:',len(self.bad_channels))
			self.last_save = time.time()
			xml = xml_handler.xml_handler(bad_channels =self.bad_channels,filename = self.filename_channels)
			xml.bad_channels2xml(multiplier = 10)
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
		elif len(self.start_epoch) >= 10: i = int(len(self.start_epoch) / 10) * int(n)
		elif int(n) < len(self.start_epoch): i =  int(n)
		else: i = len(self.start_epoch) -1
			
		if not self.e_index == i:
			self.e_index = i
			self.reset_visible()
			self.redraw_plot()


	def reset_visible(self):
		'''Check whether each bad_epoch is visible in the current plot epoch and set flag in the be accordingly.'''
		for be in self.bad_epochs:
			be.in_plot_epoch(self.start_epoch[self.e_index],self.end_epoch[self.e_index])
		for bc in self.bad_channels:
			bc.in_plot_epoch(self.start_epoch[self.e_index],self.end_epoch[self.e_index])


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
		for bc in self.bad_channels:
			if bc.visible and bc.redraw:
				self.redraw = True
				bc.redraw = False
				

	def redraw_plot(self):
		'''Redraw current plot window (because something has been removed.)'''
		plt.close(self.fig)
		for be in self.bad_epochs:
			be.set_complete_replot()
		for bc in self.bad_channels:
			bc.set_complete_replot()
		self.plot_epoch('all',offset_value = self.offset_value)


	def find_before_and_after_boundaries(self):
		'''aggregate all before and all after boundaries (relative to mouse) 
		in seperate lists of ascending order (distance from mouse).'''
		self.boundaries.sort()
		self.channel_boundaries.sort()
		self.before_boundaries, self.after_boundaries = [] , []
		self.before_ch_boundaries, self.after_ch_boundaries = [] , []

		for i,b in enumerate(self.boundaries):
			if b.x < self.event.xdata:
				self.before_boundaries.append(b)
			if b.x > self.event.xdata:
				self.after_boundaries.append(b)
		self.before_boundaries.reverse()

		for i,b in enumerate(self.channel_boundaries):
			if b.x < self.event.xdata:
				self.before_ch_boundaries.append(b)
			if b.x > self.event.xdata:
				self.after_ch_boundaries.append(b)
		self.before_ch_boundaries.reverse()
			

	def get_bad_epoch(self,boundary):
		'''return bad epoch that contains this boundary.
		boundaries should alway be contained in a bad epoch.'''
		if self.channel_mode == 'on':
			for bc in self.bad_channels:
				if boundary in bc: return bc
			return None
		for be in self.bad_epochs:
			if boundary in be:
				return be




	def find_completion_bad_epoch(self,boundary_type):
		'''find bad epoch that is closest in time with missing boundary of correct type within 2 plot epochs of mouse.
		boundary_type 		start or end, specifying boundary type that completes current boundary i.e. if a start 
							boundary is made an epoch should be searched with only an end boundary'''
		if self.channel_mode == 'on':
			if boundary_type =='end': boundaries = self.after_ch_boundaries
			elif boundary_type == 'start': boundaries = self.before_ch_boundaries
			else: return 0
		else:	
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
		if self.channel_mode == 'on': bads = self.bad_channels
		else: bads = self.bad_epochs
		for i,be in enumerate(bads):
			if be.epoch_id == epoch_id:
				index = i
				break
		print(i)
		if index > -1:
			if self.channel_mode == 'on': be = self.bad_channels.pop(index)
			else: be = self.bad_epochs.pop(index)
			print('removing following epoch')
			print(be)
			print('n bad epochs',len(self.bad_epochs))
			print('n bad channels',len(self.bad_channels))
			

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


	def annotate_bad_channel(self,annotation = ''):
		'''Set the label for the bad channel.'''
		boundaries = []
		channel = self.ch_names[self.channel_mode_index]
		for bc in self.bad_channels:
			if bc.ok and bc.visible and bc.in_bad_channel(self.mousex) and bc.channel == channel:
				boundaries.extend([bc.start,bc.end])
		dist = self.length * 1000 * 2 
		if len(boundaries) == 0: return 0
		for b in boundaries:
			if abs(b.x - self.mousex) < dist:
				closest = b
		bc = self.get_bad_epoch(closest)	
		print(self.mousex)
		print(bc)
		if annotation == 'correct' or annotation == 'incorrect':
			bc.set_correct(annotation)
		else: bc.set_annotation(annotation)

	def annotate_bad_epoch(self,annotation = ''):
		'''set the label for the bad epoch.'''
		boundaries = []
		if self.artifact_selected != None and self.annotation_type == 'corrector':
			be = self.artifact_selected
		else:
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
		if self.event.xdata > 0: x = self.event.xdata
		else: x = 0
		boundary = bad_epoch.Boundary(x,'start')
		if self.channel_mode == 'on': self.channel_boundaries.append(boundary)
		else: self.boundaries.append(boundary)
		be = self.find_completion_bad_epoch(boundary_type = 'end')
		if be:
			print('combining boundaries')
			be.set_start(boundary)
			if self.channel_mode == 'on':
				i = self.ch_names.index(be.channel)
				be.plot(channel_data = self.data[i],offset = i *self.offset_value)
		else:
			print('making new epoch')
			if self.channel_mode == 'on':
				channel = self.ch_names[self.channel_mode_index]
				self.bad_channels.append(bad_channel.Bad_channel(channel,start_boundary = boundary, pp_id = self.pp_id, coder = self.coder,exp_type = self.exp_type, bid = self.bid, block_st_sample = self.block_st_sample, epoch_id = self.make_bad_epoch_id(),offset = self.channel_mode_index *self.offset_value,annotation = self.default_annotation_channel))
			else:
				self.bad_epochs.append(bad_epoch.Bad_epoch(start_boundary = boundary, pp_id = self.pp_id, coder = self.coder,exp_type = self.exp_type, bid = self.bid, block_st_sample = self.block_st_sample, epoch_id = self.make_bad_epoch_id(),annotation = self.default_annotation))
			# self.handle_plot(force_redraw=True)	


	def handle_end(self):
		'''Create a end boundary, and either add this to closest start boundary or create new epoch.'''
		if self.event.xdata < self.data.shape[1]: x = self.event.xdata
		else: x = self.data.shape[1] - 1 
		boundary = bad_epoch.Boundary(x,'end')
		if self.channel_mode == 'on': self.channel_boundaries.append(boundary)
		else: self.boundaries.append(boundary)
		be = self.find_completion_bad_epoch(boundary_type = 'start')
		if be:
			print('combining boundaries')
			be.set_end(boundary)
			if self.channel_mode == 'on':
				i = self.ch_names.index(be.channel)
				be.plot(channel_data = self.data[i],offset = i * self.offset_value)
		else:
			if self.channel_mode == 'on':
				channel = self.ch_names[self.channel_mode_index]
				self.bad_channels.append(bad_channel.Bad_channel(channel,end_boundary = boundary, pp_id = self.pp_id, exp_type = self.exp_type, bid = self.bid, block_st_sample = self.block_st_sample, epoch_id = self.make_bad_epoch_id(),offset = self.channel_mode_index * self.offset_value,annotation = self.default_annotation_channel))
			else:
				self.bad_epochs.append(bad_epoch.Bad_epoch(end_boundary = boundary, pp_id = self.pp_id, exp_type = self.exp_type, bid = self.bid, block_st_sample = self.block_st_sample, epoch_id = self.make_bad_epoch_id(),annotation = self.default_annotation))
			# self.handle_plot(force_redraw=True)


	def handle_delete(self):
		'''Delete boundary that is closest to the mouse cursor but not further than 30 away.'''
		before, after = 100, 100
		if self.channel_mode == 'on':
			if len(self.before_ch_boundaries) > 0: before = abs(self.before_ch_boundaries[0].x - self.mousex)
			if len(self.after_ch_boundaries) > 0: after =  abs(self.after_ch_boundaries[0].x - self.mousex)
		else:
			if len(self.before_boundaries) > 0: before = abs(self.before_boundaries[0].x - self.mousex)
			if len(self.after_boundaries) > 0: after =  abs(self.after_boundaries[0].x - self.mousex)
		if before > 30 and after > 30: return 0
		if self.channel_mode == 'on':
			if before < after: boundary = self.before_ch_boundaries[0]
			if before > after: boundary = self.after_ch_boundaries[0]
		else:
			if before < after: boundary = self.before_boundaries[0]
			if before > after: boundary = self.after_boundaries[0]

		be = self.get_bad_epoch(boundary)
		if self.channel_mode == 'on' and be.channel == self.ch_names[self.channel_mode_index]: 
			self.channel_boundaries.pop(self.channel_boundaries.index(boundary))
			be.del_boundary(boundary)
		elif self.channel_mode == 'off': 
			self.boundaries.pop(self.boundaries.index(boundary))
			be.del_boundary(boundary)
		if be.empty: 
			self.delete_bad_epoch(be.epoch_id)
			self.redraw = True




	def find_next_artifact_epoch(self,direction = 'forward'):
		artifact_names = ['garbage','unk','drift','artifact']
		if direction == 'forward': epoch_index = self.e_index + 1
		if direction == 'backward': epoch_index = self.e_index - 1
		while True:
			for be in self.bad_epochs:
				if be.annotation in artifact_names:
					if self.start_epoch[epoch_index] <= be.start.x <= self.end_epoch[epoch_index]:
						if be.epoch_id != self.old_epoch_id:
							self.old_epoch_id = be.epoch_id
							return epoch_index
			if direction == 'forward': epoch_index += 1
			if direction == 'backward': epoch_index -= 1

			if direction == 'forward' and epoch_index >= len(self.start_epoch):
				print('full circle LAST BAD EPOCH LAST BAD EPOCH\n'*30)
				return len(self.start_epoch) - 1
			if direction == 'backward' and epoch_index < 0:
				print('full circle LAST BAD EPOCH LAST BAD EPOCH\n'*30)
				return 0



	def jump_to_next_artifact(self):
		new_e_index = self.find_next_artifact_epoch()
		if self.e_index != new_e_index: 
			self.e_index = new_e_index
			self.reset_visible()
			self.handle_plot(force_redraw = True)
		else: self.jump_to_next_artifact()
		
			

	def jump_to_previous_artifact(self):
		new_e_index = self.find_next_artifact_epoch('backward')
		if self.e_index != new_e_index: 
			self.e_index = new_e_index
			self.reset_visible()
			self.handle_plot(force_redraw = True)
		else: self.jump_to_previous_artifact()


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

	def toggle_view_mode(self):
		if self.view_mode == 'all':
			self.view_mode = 'butterfly'
			self.offset_value = 0
		elif self.view_mode == 'butterfly':
			self.view_mode = 'all'
			self.offset_value = 40
		self.handle_plot(force_redraw=True)

	def toggle_zoom(self,direction = 'in'):
		self.length_list = [l*self.sf for l in [10,20,40,80,100,200]]
		self.center =  self.end_epoch[self.e_index] - (self.length/2)
		if self.event.xdata != None: self.mousex = self.event.xdata
		else: self.mousex = self.center
		
		if self.length not in self.length_list:
			self.length_list.append(self.length)
			self.length_list = sorted(self.length_list)
		i = self.length_list.index(self.length)
		if direction == 'in':
			# if i == 0: return False
			self.length = self.length_list[i-1]
		if direction == 'out':
			# if i == len(self.length_list) -1: return False
			self.length = self.length_list[i+1]
		if type(direction) ==int: self.length =int(direction) * self.sf
		self.make_epoch()
		print(i,self.length)
		for i in range(len(self.start_epoch)):
			if self.artifact_selected:
				self.artifact_selected.in_plot_epoch(self.start_epoch[i],self.end_epoch[i])
				if self.artifact_selected.visible: 
					self.e_index = i
					break
			elif self.start_epoch[i] < self.mousex< self.end_epoch[i]:
				self.e_index = i
		self.reset_visible()
		self.handle_plot(force_redraw=True)
		plt.axvline(self.mousex,color='green',linestyle='--',linewidth=30,alpha=0.3)
		

	def find_channel(self,event):
		'''Create a bad channel object. Work In Progress'''
		self.event = event
		closest = 2000
		if self.offset_value== 0:ch_y = self.data[:,int(self.event.xdata)]
		elif self.event.xdata < self.start_epoch[self.e_index]: ch_y = self.ch_starty
		else: ch_y = self.ch_endy
		for i,y in enumerate(ch_y):
			delta = abs(self.event.ydata - y)
			if delta < closest: 
				closest = delta
				self.channel_mode_index = i
			

	def toggle_channel_mode(self,event):
		if event.key == ';':
			if self.channel_mode == 'on': self.channel_mode = 'off'
			else: self.channel_mode = 'on'
		else: self.channel_mode = 'on'
		if self.channel_mode == 'on': self.heog_view = 'off'
		if event.key == 'up': self.channel_mode_index+= 1
		if event.key == 'down': self.channel_mode_index-= 1
		if self.channel_mode_index < 0: self.channel_mode_index = len(self.ch_names) -1
		elif self.channel_mode_index > len(self.ch_names)-1: self.channel_mode_index = 0
		if event.key == "'":self.find_channel(event)
		self.handle_plot(force_redraw=True)
			

	def toggle_complete_channel(self):
		if self.channel_mode == 'off': return False
		channel = self.ch_names[self.channel_mode_index]
		if channel in self.complete_bad_channel:
			self.complete_bad_channel.pop(self.complete_bad_channel.index(channel))
			for bc in self.bad_channels:
				if bc.channel == channel and bc.annotation == 'all':
					self.delete_bad_epoch(bc.epoch_id)
					break
		else:
			sboundary = bad_epoch.Boundary(0,'start')
			eboundary = bad_epoch.Boundary(self.data.shape[1],'end')
			self.bad_channels.append(bad_channel.Bad_channel(channel,start_boundary = sboundary,end_boundary = eboundary, pp_id = self.pp_id, coder = self.coder,exp_type = self.exp_type, bid = self.bid, block_st_sample = self.block_st_sample, epoch_id = self.make_bad_epoch_id(),offset = self.channel_mode_index *self.offset_value,annotation = 'all'))
			self.complete_bad_channel.append(channel)

		self.handle_plot(force_redraw=True)
		

	def toggle_heog(self):
		if self.heog_view == 'on':
			self.heog_view = 'off' 
		elif self.heog_view == 'off':
			self.heog_view = 'on'
			self.channel_mode = 'off'
		self.handle_plot(force_redraw=True)

	def select_artifact(self,move = 'next'):
		'''Select a bad epoch that is visible on screen without moving mouse.'''
		print('selecting...')
		if move == 'clear':
			print('clearing')
			self.artifact_selected = None
			self.handle_plot(force_redraw=True)
			return 0

		visible_be = []
		for be in self.bad_epochs:
			if be.visible:
				visible_be.append(be)

		visible_be.sort()
		print(visible_be,'found epochs')
		if self.artifact_selected == None or move == 'first':  
			index = 0
		elif move == 'last':
			index = -1
		else:
			index = visible_be.index(self.artifact_selected)
			if move == 'next':
				index += 1
				if index == len(visible_be):  
					self.jump_to_next_artifact()
					self.select_artifact('first')
					return 0
			if move == 'back':
				index -= 1
				if index < 0: 
					self.jump_to_previous_artifact()
					self.select_artifact('last')
					return 0
		self.artifact_selected = visible_be[index]
		print(self.artifact_selected,'found artifact')
		self.handle_plot(force_redraw=True)

		
	def handle_note(self):
		# plt.close(self.fig)
		n = notes.note(windower.make_name(self.b),annotation_type='channels')
		n.edit()
		self.handle_plot(force_redraw=True)
		
	def toggle_show_complete_bad(self):
		if self.show_complete_bad: self.show_complete_bad = False
		else: self.show_complete_bad = True
		self.handle_plot(force_redraw=True)

	def on_key(self,event):
		'''Handle a key press event - links to pyplot window event manager.'''
		self.event_key = event
		self.event= event
		print(event.key)
		force_save = False
		if event.key in ['b','n']:
			self.handle_epoch_switch(event.key)
		if event.key == 'K': self.toggle_show_complete_bad()
		if event.key == 'N': self.handle_note()
		if event.key == '`':
			self.jump_to_next_artifact()
		# if event.key == 'z':
			# self.jump_to_next_artifact()
		if self.channel_mode == 'on' and event.key in self.key_dict_channel.keys():
			self.annotate_bad_channel(self.key_dict_channel[event.key])
		if self.channel_mode == 'off' and event.key in self.key_dict.keys():
			self.annotate_bad_epoch(self.key_dict[event.key])
		if event.key == ' ': force_save = True
		if event.key in string.digits:
			self.handle_epoch_jump(event.key)
		if event.key == 'p': self.purge_bad_epochs()
		if event.key == 't': self.toggle_view_mode()
		if event.key == ',': self.toggle_heog()
		if event.key == '.': self.select_artifact(move ='back')
		if event.key == '/': self.select_artifact(move ='next')
		if event.key == 'z': self.select_artifact(move ='clear')
		# if event.key == 'z': self.toggle_zoom('in')
		# if event.key == 'Z': self.toggle_zoom('out')
		if event.key == 'left': self.toggle_zoom(10)
		if event.key == 'right': self.toggle_zoom(40)
		if event.key == 'down': self.toggle_channel_mode(event)
		if event.key == 'up': self.toggle_channel_mode(event)
		if event.key == ';': self.toggle_channel_mode(event)
		if event.key == "'": self.toggle_channel_mode(event)
		if event.key == 'y': self.toggle_complete_channel()
		self.handle_plot()
		self.handle_save_xml(force_save)
		self.last_event_key = event
		if self.artifact_selected != None and not self.artifact_selected.visible:
			self.artifact_selected = None
			print('clear selected artifact')
		print(self.artifact_selected,'as')

	def purge_bad_epochs(self):
		if self.channel_mode == 'off':
			for be in self.bad_epochs:
				if not be.ok:
					if be.start: self.boundaries.pop(self.boundaries.index(be.start))
					elif be.end: self.boundaries.pop(self.boundaries.index(be.end))
					self.delete_bad_epoch(be.epoch_id)
		if self.channel_mode == 'on':
			for bc in self.bad_channels:
				if not bc.ok:
					if bc.start: self.channel_boundaries.pop(self.channel_boundaries.index(bc.start))
					elif bc.end: self.channel_boundaries.pop(self.channel_boundaries.index(bc.end))
					self.delete_bad_epoch(bc.epoch_id)
			self.find_before_and_after_boundaries()
		self.handle_plot(force_redraw=True)
				

	def on_motion(self,event):
		'''Update mouse position.'''
		self.mousex = event.xdata
		self.mousey = event.ydata


				

	def plot_epoch(self,channels = ['Fz','Cz','Pz'],offset_value = 40,show_bad_epoch = True,show_bad_channels = True):
		'''Plot a window with specified channels of eeg data.
		channels 		names or indices of eeg channels, can also be all to plot all channels present
		offset_value 	vertical distance between eeg channels
		show_bad_epoch 	whether to show the bad epochs for this time window
		'''
		self.offset_value = offset_value
		if channels == []: channels = 'all'
		if channels == 'all': channels = self.ch_names
		if type(channels[0]) == int: 
			self.channel_index= copy.deepcopy(channels)
			self.channels = [self.ch_names[i] for i in channels]
		else: 
			self.channel_index = [self.ch_names.index(n) for n in channels]
			self.channels = channels

		# print(len(self.channel_index),len(self.channels),self.data.shape,self.end_epoch[-1])
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
		bad_indices = [self.ch_names.index(name) for name in self.complete_bad_channel]
		for i in self.channel_index: 
			# plot a channel
			if i in bad_indices: 
				if self.show_complete_bad:
					plt.plot(yx,y[i,:]+offset,linewidth = 6,color='black',alpha =0.2)
			elif self.channel_mode == 'on' and i != self.channel_mode_index:
				plt.plot(yx,y[i,:]+offset,linewidth = 0.9,color=clist[ci],alpha =0.1)
			elif self.heog_view == 'on' and self.ch_names[i] not in ['F7','F8']:
				plt.plot(yx,y[i,:]+offset,linewidth = 0.9,color=clist[ci],alpha =0.1)
			else:
				plt.plot(yx,y[i,:]+offset,linewidth = 0.9,color=clist[ci])
			ci +=1
			if ci == len(clist):ci =0
			# create coordinates for channel names before and after channel plot
			self.ch_starty[i] = y[i,0] + offset - 10 # subtract 10 to center name in y dimension
			self.ch_endy[i] = y[i,-1] + offset - 10 # subtract 10 to center name in y dimension
			# offset between channels
			offset += offset_value 

		#Set plot dimensions
		if offset_value == 0:
			plt.ylim((-500,500))
		else: plt.ylim((-80,(len(self.channel_index) + 2) * 40))
		plt.xlim(start_index -50,start_index + self.length+50)


		#Plot channel name for each channel in the same color
		ci = 0
		for i,y in enumerate(self.ch_starty):
			if self.channel_mode == 'on' and i != self.channel_mode_index:
				plt.annotate(self.channels[i],xy=(start_index-30,y),color = clist[ci],fontsize=18,alpha = 0.1)
			else:
				plt.annotate(self.channels[i],xy=(start_index-30,y),color = clist[ci],fontsize=18)
			ci += 1
			if ci == len(clist): ci = 0 
		ci = 0
		for i,y in enumerate(self.ch_endy):
			if self.channel_mode == 'on' and i != self.channel_mode_index:
				plt.annotate(self.channels[i],xy=(end_index + 20,y),color=clist[ci],fontsize=18,alpha=0.1)
			else:
				plt.annotate(self.channels[i],xy=(end_index + 20,y),color=clist[ci],fontsize=18)
			ci += 1
			if ci == len(clist): ci = 0 

		#plot bad epochs
		if show_bad_epoch:
			for be in self.bad_epochs:
				if be == self.artifact_selected: be.plot(selected = True)
				else: be.plot()
		if show_bad_channels:
			for bc in self.bad_channels:
				i = self.ch_names.index(bc.channel)
				if bc.annotation != 'all':
					bc.plot(channel_data = self.data[i],offset = i*offset_value)
		plt.grid(alpha=0.2,color='tan',lw=2)

		ypos = plt.ylim()[1] - 100
		if self.e_index == len(self.start_epoch) - 1:
			plt.annotate('END',xy=(self.start_epoch[-1]+400, ypos), color ='red',fontsize=70,alpha = .4)
		elif self.e_index == 0:
			plt.annotate('START',xy=(self.start_epoch[0]+800, ypos), color ='blue',fontsize=50,alpha=.4)

		if hasattr(self.b,'blink_peak_sample') and self.b.nblinks != 'NA':
			for blink in self.b.blink_peak_sample:
				if self.end_epoch[self.e_index] > blink/10 > self.start_epoch[self.e_index]:
					plt.axvline(blink/10,color='blue',linestyle='--',linewidth=3,alpha=0.3)
		else: print('no blinks')
			



'''
	def find_event(self,event):

		for k in self.event_dict.keys():
			if abs(k - event.xdata) < 15:
				return k
		return None


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
'''
