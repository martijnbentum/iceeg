from matplotlib import pyplot as plt

def enter_axes(event,self):
	'''Check in which subplot (component) the cursor is located.'''
	for i,subplot in enumerate(self.ica_plot.axes):
		if event.inaxes == subplot: 
			print('found subplot: ',i)
			self.eog_index = i
			self.current_subplot = subplot

def leave_axes(event,self):
	'''Check whether cursor is not located within any subplot (component).'''
	self.eog_index = None
	self.current_subplot = None

def on_click(event,self):
	'''Handle mouse click and perform annotation when relevant.
	if cursor not on subplot, no action is performed.
	if left click, add a component
	if right click, remove a component
	if scroll click, save and exit
	'''
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
	
def plot(self, ncomponents = 25):
	'''Plot the ica topography for the components.
	ncomponents 		number of components to plot.
	'''
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
