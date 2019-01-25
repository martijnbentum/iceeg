import block
import eog
from matplotlib import pyplot as plt
import mne
import mne.preprocessing.ica as ica
import path


def fit(self, reject_artifacts= True,reject_channels = [], block_indices = []):
	'''Fit ica on block object.
	EEG data is bandpassed filtered on 1-30 Hz, the ica solution can be used on data
	without or different bandpass filter see:
	Winkler et al.
	On the influence of high-pass filtering on ICA-based artifact reduction in EEG-ERP
	WIP: extend to be able to set ica approach (with or without artifact rejection, filename)
	
	both_reject_all 	whether to fit ica with and without rejected artifacts
	'''
	self.ica_rejected_channels = reject_channels
	self.reject_artifacts = reject_artifacts
	if hasattr(self,'raw') and self.raw.info['highpass'] < 1: self.eeg_load = False
	if not self.eeg_loaded: 
		try:self.load_eeg_data(freq = [1,30], block_indices = block_indices)
		except: return 0
	self.raw.info['bads'] = self.ica_rejected_channels
	self.ica = ica.ICA()
	if type(self) == block.block and self.artifacts != 'NA':
		self.load_artifacts()
		print('Setting artefact annotation to mne format')
		self.raw.annotations = mne.Annotations(self.start_artifacts,self.duration_artifacts,'BAD')
	if self.reject_artifacts:
		print('Excluding artefacts and fitting ica')
		identifier = '_no-artifact-ica.fif'
		self.ica_filename = path.ica_solutions + self.make_name() + identifier
	else:
		self.ica_filename = path.ica_solutions + self.make_name() + '_all-data-ica.fif'

	self.ica.fit(self.raw,reject_by_annotation = self.reject_artifacts)
	self.ica.save(self.ica_filename)
	self.create_eog()


def load(self, rejected_artifacts, filename_ica = ''):
	self.ica_rejected_artifacts = rejected_artifacts
	if filename_ica == '':
		name = self.make_name()
		self.ica_filename = path.ica_solutions + name + '_no-artifact-ica.fif'
		
	self.ica = mne.preprocessing.read_ica(self.ica_filename) 
	self.eog_filename = self.ica_filename.replace('ica.fif','eog.xml')
	self.eog = eog.load(self.eog_filename) #not all eog corr are present
	if not self.eog and hasattr(self,'ica'):
		self.create_eog()
	self.ica.exclude = self.eog.comps
	if hasattr(self,'raw'): self.raw.info['bads'] = self.eog.rejected_channels


def enter_axes(event,self):
	for i,subplot in enumerate(self.ica_plot.axes):
		if event.inaxes == subplot: 
			print('found subplot: ',i)
			self.eog_index = i
			self.current_subplot = subplot

def leave_axes(event,self):
	self.eog_index = None
	self.current_subplot = None

def on_click(event,self):
	color, close = 'blue',False
	if self.current_subplot == None:
		print('mouse not on component, doing nothing')
		return False
	if event.button == 1: # left mouse button
		print('add: ',self.eog_index)
		self.eog.add_comp(self.eog_index)
		color = 'red'
	elif event.button == 2: close = True # scroll wheel
	elif event.button == 3: # right mouse button 
		print('delete: ',self.eog_index)
		self.eog.delete_comp(self.eog_index)
		color = 'white'
	if self.eog_index in self.eog.veog_comps: marker,size = '*',18
	else: marker,size = '<',12
	self.current_subplot.plot(.4,.5,marker=marker,color = color,markersize=size)
	self.current_subplot.axvline(x=-0.6,ymin = 0.15,ymax = .93,linewidth=3,color=color)
	self.eog.ica_checked = True
	if not close: self.ica_plot.canvas.draw()
	self.eog.eog2xml()
	self.eog.write()
	if close: plt.close()
	print(self.eog.print_xml())
	return True
	
def plot(self, ncomponents = 20, selected_and_deleted = False):
	if not hasattr(self,'ica'): self.load_ica()
	self.ica.exclude = self.eog.comps
	self.ica_plot = self.ica.plot_components(range(ncomponents))
	ica_type = 'BLOCK' if 'bid' in self.eog.name else 'SESSION'
	self.ica_plot.canvas.set_window_title(self.eog.name + '   ' + ica_type)
	self.ica_plot.canvas.mpl_connect('axes_enter_event', lambda event: enter_axes(event,self))
	self.ica_plot.canvas.mpl_connect('axes_leave_event',lambda event: leave_axes(event,self))
	self.ica_plot.canvas.mpl_connect('button_press_event',lambda event: on_click(event,self))
	color = 'green'
	for i,subplot in enumerate(self.ica_plot.axes):
		veog,heog = self.eog.veog_scores_all[i].round(2), self.eog.heog_scores_all[i].round(2)
		print(veog,heog,i)
		subplot.text(x=-.65,y=-.61,s=str(abs(veog))[1:],color = color,alpha = abs(float(veog)),size =15)
		subplot.text(x=.3,y=-.61,s=str(abs(heog))[1:],color = color,alpha = abs(float(heog)),size =15)
		if i in self.eog.comps:
			if i in self.eog.veog_comps: marker,size = '*',18
			else: marker,size = '<',12
			subplot.plot(.4,.5,marker=marker,color = 'orangered',markersize=size)
			# subplot.axhline(y=0.7,linewidth=3,color='red')
			subplot.axvline(x=-0.6,ymin = 0.15,ymax = .93,linewidth=3,color='red')

def plot_overlay(self,nplots = 10):
	if not type(block.block): 
		print('cannot plot overlay for session, blink start end times are not defined for session data')
		return
	if not hasattr(self,'nblinks') or self.nblinks == 0: 
		print('no blinks, doing nothing')
	if self.nblinks < nplots: nplots = self.nblinks
	ii= list(range(nplots))
	[self.ica.plot_overlay(self.raw,start = self.blink_start[i], stop=self.blink_end[i]) for i in ii]




