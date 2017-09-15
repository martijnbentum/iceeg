import bad_epoch
import copy
from matplotlib import pyplot as plt
import numpy as np
import os
import pygame
from scipy import stats

class ac:
	def __init__(self,g,length = 10,decimate = 5,overlap = False,filename = '',load_data = True):
		'''Interface to easily annotate eeg signal
		g 			garbage stats object, class defined in garbage_collection.py
		length 		duration in seconds of an epoch in the interface
		decimate 	take sample every n samples, speeds up plotting
		overlap 	whether plot windows should overlap
		filename 	specify xml file that is loaded for bad_epochs, default is to generate filename based on block info
		load_data 	surpress previously generated bad_epochs (create new annotation), old versions are moved to OLD 
					directory in artifacts folder.
		'''

		self.key_dict = {'a':'alpha','g':'garbage','m':'movement','u':'unk','e':'blink'}
		self.g = g
		self.set_info()
		self.check_filename(filename)
		self.length = int(float(length) * 1000)
		self.decimate = decimate
		self.e_index = 0
		self.event_dict = {}
		self.overlap = overlap
		self.bad_epochs = []
		self.make_epoch()
		self.plot_epoch('all',self.decimate)
		self.last_bad_epoch_id = 0
		self.redraw = False
		self.run()


	def check_filename(self,filename):
		if os.path.isfile(filename): self.filename = filename
		else:
			print('Auto generating filename based on block information.')
			filename = path.artifacts +'pp' + str(self.pp_id) + '_exp-' + self.exp_type + '_bid' + str(self.bid) + '.xml'
		elif os.path.isfile(
		else: print('File',filename,'not found.')
			

	def set_info(self):
		'''Set experimental info (participant id, experiment type, etc.) to current object.'''
		b = self.g.block
		self.pp_id, self.exp_type, self.bid,self.block_st_sample = b.pp_id, b.exp_type, b.bid, b.st_sample


	def make_epoch(self):
		'''Create a start and end numpy array with start and end times of plot windows.
		Length (specified in seconds set the length of the window, overlap specifies whether the plot
		windows should overlap half of their length.'''
		if self.overlap:
			# If overlap is true make it overlap for half of the window
			self.start_epoch= np.arange(0,self.g.duration_sample,int(self.length/2))
		else:
			self.start_epoch= np.arange(0,self.g.duration_sample,self.length)
		self.end_epoch= self.start_epoch+ self.length
		# last epoch can only last until end data
		self.end_epoch[-1] = self.g.duration_sample


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


	def reset_visible(self):
		'''Check whether each bad_epoch is visible in the current plot epoch and set flag in the be accordingly.'''
		for be in self.bad_epochs:
			be.in_plot_epoch(self.start_epoch[self.e_index],self.end_epoch[self.e_index])


	def handle_plot(self):
		'''Check whether something has changed that requires a redraw of the plot window. Peform redraw when necessary.'''
		self.fig.canvas.draw()
		self.check_redraw()
		if self.redraw:
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
		self.plot_epoch('all',decimate = self.decimate)


	def find_closest_epoch(self):
		'''Find the bad epoch that is closest to the current position of the mouse (before and after.'''
		self.before, self.after = self.length * 2 * 1000, self.length * 2 * 1000
		self.before_epoch, self.after_epoch = None, None
		self.before_boundary, self.after_boundary= None, None
		for be in self.bad_epochs:
			if be.end != None:
				if be.end.x - self.event.xdata < 0: 
					if abs(be.end.x - self.event.xdata) < self.before: 
						self.before = abs(be.end.x - self.event.xdata)
						self.before_epoch = be
						self.before_boundary = be.end
				else:
					if abs(be.end.x - self.event.xdata) < self.after: 
						self.after = abs(be.end.x - self.event.xdata)
						self.after_epoch = be
						self.after_boundary = be.end
			if be.start != None:
				if be.start.x - self.event.xdata < 0: 
					if abs(be.start.x - self.event.xdata) < self.before: 
						self.before = abs(be.start.x - self.event.xdata)
						self.before_epoch = be
						self.before_boundary = be.start
				else:
					if abs(be.start.x - self.event.xdata) < self.after: 
						self.after = abs(be.start.x - self.event.xdata)
						self.after_epoch = be
						self.after_boundary = be.start


	def delete_bad_epoch(self,epoch_id):
		'''Delete a bad_epoch object based on epoch_id.'''
		index = -1
		for i,be in enumerate(self.bad_epochs):
			if be.epoch_id == epoch_id:
				index = i
				break
		if index > -1:
			self.bad_epochs.pop(index)
			

	def make_bad_epoch_id(self):
		'''Create an unique integer id.'''
		self.last_bad_epoch_id += 1
		return self.last_bad_epoch_id


	def annotate_bad_epoch(self,annotation = ''):
		'''Set the label for the bad epoch.'''
		for be in self.bad_epochs:
			if be.ok and be.visible and be.in_bad_epoch(self.mousex):
				print(self.mousex)
				print(be)
				print(be.in_bad_epoch(self.mousex))
				be.set_annotation(annotation)


	def handle_start(self):
		'''Create a start boundary, and either add this to closest end boundary or create new epoch.'''
		boundary = bad_epoch.Boundary(self.event.xdata,'start')
		if not self.after_epoch == None and self.after_epoch.start == None:
			self.after_epoch.set_start(boundary)
		else:
			self.bad_epochs.append(bad_epoch.Bad_epoch(start_boundary = boundary, pp_id = m.pp_id, exp_type = m.exp_type, block_st_sample = m.block_st_sample, epoch_id = self.make_epoch_id))
			

	def handle_end(self):
		'''Create a end boundary, and either add this to closest start boundary or create new epoch.'''
		boundary = bad_epoch.Boundary(self.event.xdata,'end')
		if not self.before_epoch == None and self.before_epoch.end== None:
			self.before_epoch.set_end(boundary)
		else:
			self.bad_epochs.append(bad_epoch.Bad_epoch(start_boundary = boundary, pp_id = m.pp_id, exp_type = m.exp_type, block_st_sample = m.block_st_sample, epoch_id = self.make_epoch_id))

	def handle_delete(self):
		'''Delete boundary that is closest to the mouse cursor but not further than 30 away.'''
		if self.before < self.after and self.before < 30:
			self.before_epoch.del_boundary(self.before_boundary)
			if self.before_epoch.empty: 
				self.delete_bad_epoch(self.before_epoch.epoch_id)
				self.before_epoch = None
				self.redraw = True

		elif self.before > self.after and self.after < 30:
			self.after_epoch.del_boundary(self.after_boundary)
			if self.after_epoch.empty: 
				self.delete_bad_epoch(self.after_epoch.epoch_id)
				self.after_epoch = None
				self.redraw = True
		self.handle_plot


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


	def on_click(self,event):
		'''Handle click event - links to pyplot window event manager.'''
		self.event = event
		if self.event.xdata != None and self.event.ydata != None:
			print(self.event.xdata, self.event.ydata)
			self.find_closest_epoch()
			if self.event.xdata < self.start_epoch[self.e_index]  or self.event.xdata > self.end_epoch[self.e_index]:
				self.bad_channel()
			elif self.event.button == 1: # left mouse button
				self.handle_start()
			elif self.event.button == 2: # scroll wheel
				self.handle_delete()
			elif self.event.button == 3: # right mouse button
				self.handle_end()
		self.last_event = event
		self.handle_plot()
         

	def on_key(self,event):
		'''Handle a key press event - links to pyplot window event manager.'''
		self.event_key = event
		print(event.key)
		if event.key in ['b','n']:
			self.handle_epoch_switch(event.key)
		if event.key in ['a','m','g','u']:
			self.annotate_bad_epoch(self.key_dict[event.key])
		self.handle_plot()
		self.last_event_key = event


	def on_motion(self,event):
		'''Update mouse position.'''
		self.mousex = event.xdata
		self.mousey = event.ydata


				

	def plot_epoch(self,channels = ['Fz','Cz','Pz'],decimate = 5,offset_value = 40,show_bad_epoch = True):
		'''Plot a window with specified channels of eeg data.
		channels 		names or indices of eeg channels, can also be all to plot all channels present
		decimate 		data reduction to increase plotting speed (decimate 5, keep 1 in 5 data points)
		offset_value 	vertical distance between eeg channels
		show_bad_epoch 	whether to show the bad epochs for this time window
		'''
		if channels == []: channels = 'all'
		if channels == 'all': channels = self.g.ch_names
		if type(channels[0]) == int: 
			self.channel_index= copy.deepcopy(channels)
			self.channels = [g.ch_names[i] for i in channels]
		else: 
			self.channel_index = [self.g.ch_names.index(n) for n in channels]
			self.channels = channels

		#Create figure
		self.fig, self.ax = plt.subplots(num=None, figsize=(21, 11), dpi=80, facecolor='w', edgecolor='k')
		self.fig.tight_layout()

		#Create handles that connect to the pyplot for mouse clicks, key presses and mouse motion.
		self.cidm = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.cidk = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


		# data is lowpass filtered at 30 Hz, nyquist = 60 Hz and sf = 1000, so
		# decimate should not exceed 15, 10 speeds up plotting nicely
		if decimate > 15: decimate = 15 
		# yx = np.arange(0,self.g.data.shape[1],decimate)
		start_index,end_index = self.start_epoch[self.e_index], self.end_epoch[self.e_index]
		yx = np.arange(start_index,end_index,decimate)
		y = self.g.data[:,yx]
			
		offset = 0
		self.ch_starty = np.zeros(len(self.channel_index))
		self.ch_endy = np.zeros(len(self.channel_index))
		for i in self.channel_index: 
			# plot a channel
			plt.plot(yx,y[i,:]+offset,linewidth = 0.8)
			# create coordinates for channel names before and after channel plot
			self.ch_starty[i] = y[i,0] + offset - 10 # subtract 10 to center name in y dimension
			self.ch_endy[i] = y[i,-1] + offset - 10 # subtract 10 to center name in y dimension
			# offset between channels
			offset += offset_value 

		#Set plot dimensions
		plt.ylim((-80,(len(self.channel_index) + 2) * 40))
		plt.xlim(start_index -500,start_index + self.length+500)


		#Plot channel name for each channel in the same color
		clist = 'b,g,r,m,y,c'.split(',')
		ci = 0
		for i,y in enumerate(self.ch_starty):
			plt.annotate(self.channels[i],xy=(start_index-200,y),color = clist[ci])
			ci += 1
			if ci == len(clist): ci = 0 
		ci = 0
		for i,y in enumerate(self.ch_endy):
			plt.annotate(self.channels[i],xy=(end_index + 20,y),color=clist[ci])
			ci += 1
			if ci == len(clist): ci = 0 

		#plot bad epochs
		if show_bad_epoch:
			for be in self.bad_epochs:
				be.plot()



'''
	def find_event(self,event):

		for k in self.event_dict.keys():
			if abs(k - event.xdata) < 15:
				return k
		return None

'''
