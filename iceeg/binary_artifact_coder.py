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
import xml_cnn

class bac:
	def __init__(self,b,length = 4,corrector= 'martijn',filename = '',load_xml = True):
		'''Interface to easily annotate eeg signal
		prediction 	filename for np array with classification for each segment
		length 		duration in seconds of an epoch in the interface
		coder 		Name of the coder (if automatic it should be computer)
		filename 	specify xml file that is loaded for bad_epochs, default is to generate filename based on block info
		'''
		self.show_label = False
		self.show_correctness = False 
		self.show_perc = False

		self.key_dict = {'1':'clean','2':'artifact','c':'channel','u':'signal_unclear','d':'ac-balance_unclear','3':'back'}
		self.redraw = False
		self.b = b
		self.w = windower.Windower(b,sf =100)
		self.set_info()
		if filename == '' or type(filename) != str: self.filename = path.artifact_data_all_pp + self.w.name + '_artifacts-sf100.xml'
		else: self.filename = filename
		self.length = int(float(length) * 100)
		self.before, self.after = int(self.length /2), int(self.length /2) 
		self.e_index = 0
		self.event_dict = {}
		self.corrector=corrector 
		self.bad_epochs = []
		self.boundaries= []
		self.load_from_xml(self.filename)
		self.load_data()
		self.last_save = time.time()
		self.make_epoch()
		print(self.bad_epochs)
		self.plot_epoch('all')
		self.reset_visible()
		self.handle_plot(True)
		self.redraw = False
		self.run()

	def load_data(self):
		self.fn_data = windower.make_name(self.b)
		self.data = utils.load_100hz_numpy_block(self.fn_data) * 10 ** 6
		self.all_ch_names = utils.load_ch_names()
		self.remove_ch = ['VEOG','HEOG','TP10_RM','STI 014','LM','Fp2']
		self.ch_names = [ch for ch in self.all_ch_names if ch not in self.remove_ch]
		self.ch_index = [self.all_ch_names.index(ch) for ch in self.ch_names]
		self.data = self.data[self.ch_index,:]
		self.ch_index = [self.ch_names.index(ch) for ch in self.ch_names]


	def load_from_xml(self,filename = ''):
		if os.path.isfile(filename): self.filename = filename
		elif os.path.isfile(path.artifact_data_all_pp+ filename): self.filename = path.artifact_data_all_pp+ filename
		elif not os.path.isfile(self.filename): raise ValueError(self.filename,'not a file, please specify xml filename.')
		print(self.filename)
		xml = xml_cnn.xml_cnn(filename = self.filename)
		xml.read_bad_epoch_xml(self.filename)
		self.bad_epochs = xml.xml2bad_epochs()
		for be in self.bad_epochs:
			be.note = ''
			be.set_corrector( self.corrector)
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
			xml = xml_cnn.xml_cnn(bad_epochs=self.bad_epochs,filename=self.filename)
			xml.bad_epochs2xml()
			xml.write_update()
			

	def set_info(self):
		'''Set experimental info (participant id, experiment type, etc.) to current object.'''
		self.exp_dict = utils.exptype2int
		b = self.b
		self.pp_id, self.exp_type, self.bid,self.block_st_sample = b.pp_id, b.exp_type, b.bid, b.st_sample


	def make_epoch(self):
		'''Create a start and end numpy array with start and end times of plot windows.
		Length (specified in seconds set the length of the window,
		'''
		self.start_epoch= [b.st_sample - self.before for b in self.bad_epochs]
		self.end_epoch= [b.et_sample + self.after for b in self.bad_epochs]


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
			be.visible = False
		self.bad_epochs[self.e_index].visible = True


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


	def add_zeros(self,goal_length,number):
		l = len(str(number))
		nzeros = goal_length - l
		if nzeros > 0: return '0' * nzeros + str(number)
		else: return str(number)
			

	def set_correctness_bad_epoch(self,annotation):
		be = self.bad_epochs[self.e_index]
		print(be)
		if annotation == be.annotation:
			be.set_correct('correct')
		else: be.set_correct('incorrect')

	def add_note(self,key):
		be = self.bad_epochs[self.e_index]
		be.add_note(self.key_dict[key])
		


	def on_key(self,event):
		'''Handle a key press event - links to pyplot window event manager.'''
		self.event_key = event
		print(event.key)
		force_save,force_redraw = False, False
		if event.key in ['b','n']:
			self.handle_epoch_switch(event.key)
		if event.key in self.key_dict.keys():
			if event.key in ['1','2']:
				self.set_correctness_bad_epoch(self.key_dict[event.key])
				self.handle_epoch_switch('n')
			elif event.key == '3':
				self.handle_epoch_switch('b')
			else: 
				self.add_note(event.key)
				force_redraw = True
		if event.key == ' ': force_save = True
		if event.key == 'p':
			print('purging note:',self.bad_epochs[self.e_index])
			self.bad_epochs[self.e_index].note = ''
			force_redraw = True
		# self.key_dict = {'1':'clean','2':'artifact','c':'channel','u':'signal_unclear','d':'ac-balance_unclear','3':'back'}
		if event.key == 'i':
			self.show_label = not self.show_label
			self.show_correctness = not self.show_correctness
			self.show_perc = self.show_correctness
			force_redraw = True
		if event.key == '5':
			self.show_perc = not self.show_perc
			force_redraw = True
		
		# if event.key in string.digits:
		# self.handle_epoch_jump(event.key)
		self.handle_plot(force_redraw)
		self.handle_save_xml(force_save)
		self.last_event_key = event
				

	def plot_epoch(self,channels = ['Fz','Cz','Pz'],offset_value = 40,show_bad_epoch=True,show_label=None,show_correctness=None,show_perc = None):
		'''Plot a window with specified channels of eeg data.
		channels 		names or indices of eeg channels, can also be all to plot all channels present
		offset_value 	vertical distance between eeg channels
		show_bad_epoch 	whether to show the bad epochs for this time window
		show_label 		show cnn predicted label
		show_correctnes show whether corrector agrees
		'''
		if show_label == None: show_label = self.show_label
		if show_correctness == None: show_correctness= self.show_correctness
		if show_perc== None: show_perc= self.show_perc

		be = self.bad_epochs[self.e_index]

		if channels == []: channels = 'all'
		if channels == 'all': channels = self.ch_names
		if type(channels[0]) == int: 
			self.channel_index= copy.deepcopy(channels)
			self.channels = [g.ch_names[i] for i in channels]
		else: 
			self.channel_index = [self.ch_names.index(n) for n in channels]
			self.channels = channels

		#Create figure
		# self.fig, self.ax = plt.subplots(num=None, figsize=(21, 11), dpi=80 )
		self.fig, self.ax = plt.subplots(num=None, figsize=(20, 9), dpi=80 )
		self.fig.tight_layout()
		self.fig.patch.set_facecolor('ivory')
		self.ax.set_facecolor('ivory')
		self.fig.canvas.set_window_title('pp'+ str(self.pp_id) + '  ' + self.exp_type + ' b' + str(self.bid) + '    nbe' + str(len(self.bad_epochs)))

		#Create handles that connect to the pyplot for mouse clicks, key presses and mouse motion.
		self.cidk = self.fig.canvas.mpl_connect('key_press_event', self.on_key)


		start_index,end_index = self.start_epoch[self.e_index], self.end_epoch[self.e_index]
		if end_index > self.data.shape[1]: end_index = self.data.shape[1]
		yx = np.arange(start_index,end_index)
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
		plt.xlim(start_index -500,start_index + self.length+500)

	

		#Plot channel name for each channel in the same color
		ci = 0
		for i,y in enumerate(self.ch_starty):
			plt.annotate(self.channels[i],xy=(start_index-50,y),color = clist[ci],fontsize=18)
			ci += 1
			if ci == len(clist): ci = 0 
		ci = 0
		for i,y in enumerate(self.ch_endy):
			plt.annotate(self.channels[i],xy=(end_index + 20,y),color=clist[ci],fontsize=18)
			ci += 1
			if ci == len(clist): ci = 0 

		#plot bad epochs
		if show_bad_epoch:
			be.plot(False,False)
		plt.grid(alpha=0.2,color='tan',lw=2)
		if self.e_index == len(self.start_epoch) - 1:
			plt.annotate('END',xy=(self.end_epoch[-1] + 40, 980), color ='red',fontsize=70)
		elif self.e_index == 0:
			plt.annotate('START',xy=(self.end_epoch[0]-800, 980), color ='blue',fontsize=50)

		annotation = ''
		if be.correct == 'incorrect':
			if be.annotation == 'artifact': annotation = 'CLEAN'
			elif be.annotation == 'clean': annotation = 'ARTIFACT'
		elif be.correct == 'correct':
			if be.annotation == 'artifact': annotation = 'ARTIFACT'
			elif be.annotation == 'clean': annotation = 'CLEAN'

		plt.annotate(annotation,(end_index +160,800),fontsize=35)

		plt.annotate('NOTES:',(end_index +90,500), fontsize=25)
		h = 450
		for n in be.note.split(','):
			plt.annotate(n,(end_index +100,h), fontsize=20)
			h -= 70
		
		perc = ''
		if show_perc:
			if be.annotation == 'artifact':
				perc = str(1 - float(be.perc_clean))[:4]
			elif be.annotation == 'clean':
				perc = str( float(be.perc_clean))[:4]
		plt.annotate(perc,(start_index-300,600),fontsize=35)

		print('l',show_label,'c',show_correctness)
		if show_label:
			plt.annotate( be.annotation,(start_index-300,800),fontsize=35,color = 'purple')
		if show_correctness:
			if be.correct == 'correct': color = 'green'
			else: color = 'red'
			plt.annotate( be.correct,(start_index-300,300),fontsize=35,color = color)


