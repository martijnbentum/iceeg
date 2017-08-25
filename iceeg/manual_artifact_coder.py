from matplotlib import pyplot as plt
import numpy as np
import pygame
from scipy import stats

class ac:
	def __init__(self,g,length = 10,dec = 5,overlap = False):
		self.g = g
		self.length = int(float(length) * 1000)
		self.dec = dec
		self.e_index = 0
		self.overlap = overlap
		self.make_epoch(self.length)
		self.plot_epoch('all',self.dec)
		# self.run()

	def run(self):
		a = input('b/n:')
		print('you provided:',a)
		self.handle_epoch_switch(a)
		if a == 'exit': return 0
		self.run()

	def handle_epoch_switch(self,a):
		if a == 'n':
			self.e_index += 1
			if self.e_index == len(self.start_epoch): self.e_index = 0
		if a == 'b':
			self.e_index -= 1
			if self.e_index < 0: self.e_index = len(self.start_epoch) - 1
		if a == 'n' or a == 'b':
			plt.close(self.fig)
			self.plot_epoch('all',dec = self.dec)


	def make_epoch(self,length):
		if self.overlap:
			self.start_epoch= np.arange(0,self.g.duration_sample,int(self.length/2))
		else:
			self.start_epoch= np.arange(0,self.g.duration_sample,self.length)
		self.end_epoch= self.start_epoch+ self.length
		# last epoch can only last until end data
		self.end_epoch[-1] = self.g.duration_sample


	def onclick(self,event):
		if event.xdata != None and event.ydata != None:
			 print(event.xdata, event.ydata)
			 plt.axvline(event.xdata,color='blue',linestyle='-')
			 self.fig.canvas.draw()
         
	def on_key(self,event):
		print(event.key)
		if event.key in ['b','n']:
			self.handle_epoch_switch(event.key)
		if event.key == 'a':
			# resp = input('provide annotation:')
			print('annotate')
			# self.run()


	def plot_epoch(self,channels = ['Fz','Cz','Pz'],dec = 10,show_bad_epoch = True):
		if channels == []: channels = 'all'
		if channels == 'all': channels = self.g.ch_names
		if type(channels[0]) == int: 
			channel_index= copy.deepcopy(channels)
			channels = [g.ch_names[i] for i in channels]
		else: channel_index = [self.g.ch_names.index(n) for n in channels]
		# plt.figure(str(self.g.marker))

		self.fig, self.ax = plt.subplots(num=None, figsize=(21, 11), dpi=80, facecolor='w', edgecolor='k')
		self.fig.tight_layout()

		self.cidm = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cidk = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

		# ax.axvspan(8, 14, alpha=0.5, color='red') 


		# data is lowpass filtered at 30 Hz, nyquist = 60 Hz and sf = 1000, so
		# decimate should not exceed 15, 10 speeds up plotting nicely
		if dec > 15: dec = 15 
		# yx = np.arange(0,self.g.data.shape[1],dec)
		start_index,end_index = self.start_epoch[self.e_index], self.end_epoch[self.e_index]
		yx = np.arange(start_index,end_index,dec)
		y = self.g.data[:,yx]
			
		offset = 0
		ch_starty = np.zeros(len(channel_index))
		ch_endy = np.zeros(len(channel_index))
		for i in channel_index: 
			plt.plot(yx,y[i,:]+offset,linewidth = 0.8)
			ch_starty[i] = y[i,0] + offset
			ch_endy[i] = y[i,-1] + offset
			offset += 40
		plt.ylim((-80,(len(channel_index) + 2) * 40))
		plt.xlim(start_index -500,start_index + self.length+500)

		clist = 'b,g,r,m,y,c'.split(',')
		ci = 0
		for i,y in enumerate(ch_starty):
			plt.annotate(channels[i],xy=(start_index-200,y - 10),color = clist[ci])
			ci += 1
			if ci == len(clist): ci = 0 
		ci = 0
		for i,y in enumerate(ch_endy):
			plt.annotate(channels[i],xy=(end_index + 20,y - 10),color=clist[ci])
			ci += 1
			if ci == len(clist): ci = 0 

		if show_bad_epoch:
			start = [i for i in self.g.start_snippets[self.g.epoch_bi] if i > start_index and i < end_index]
			end = [i for i in self.g.end_snippets[self.g.epoch_bi] if i > start_index and i < end_index]
			[plt.axvline(s,color='tomato',linestyle='-',linewidth=1,alpha=0.1) for s in start]
			[plt.axvline(e,color='tomato',linestyle='--',linewidth=1,alpha=0.1) for e in end]
		 # plt.legend(channels)





