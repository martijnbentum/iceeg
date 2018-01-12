from matplotlib import pyplot as plt
import copy

class Bad_epoch:
	'''Object to store information about streches of EEG data that have artifacts.
	Contains boundaries (object) of type start and end with start sample with respect to start block
	Contains info about participant id experiment type block number and start sample of block
	'''
		
	def __init__(self,start_boundary = None,end_boundary = None,annotation = 'garbage',coder = 'martijn',color = 'blue', pp_id = None, exp_type = None, block_st_sample = None, bid = None, epoch_id = None, visible = True, correct = 'correct',perc_clean = -9,epoch_ids = '',block_et_sample = None):
		'''Information about stretch of eeg data containing artefacts.
		start_boundary 		boundary object with start sample number of type start
		end_boundary 		boundary object with end sample number of type end
		annotation 			label for the type of artefact
		color 				color of epoch in plot
		'''
		self.color_dict = {'heog':'green','jump':'pink','drift':'yellow','alpha':'green','garbage':'blue','movement':'cyan','unk':'red','blink':'magenta','ch-jump':'orange','artifact':'blue','clean':'white'}
		self.coder = coder
		self.visible = visible
		self.start = start_boundary
		self.end = end_boundary
		self.ok = False
		self.plotted = False
		self.color = color
		self.set_info()
		self.correct = correct
		self.perc_clean = perc_clean
		self.set_annotation( annotation )
		self.epoch_id = epoch_id
		self.check_boundaries()
		self.epoch_ids = epoch_ids
		self.block_et_sample = block_et_sample
		self.note = ''
		self.corrector = ''
		# print( (pp_id), exp_type, (bid), (block_st_sample), self.epoch_id)
		try: self.pp_id, self.exp_type, self.bid, self.block_st_sample = int(pp_id), exp_type, int(bid), int(block_st_sample)
		except: 
			self.pp_id, self.exp_type, self.bid, self.block_st_sample = pp_id, exp_type, bid, block_st_sample
			print('missing id information to identify this bad epoch. If you are annotating, this is bad!')
		self.redraw = False

	def __str__(self):
		m = 'Epoch id:\t\t'+str(self.epoch_id) + '\n'
		if self.ok:
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


	def __eq__(self,other):
		if type(other) != Bad_epoch: return False
		return self.start == other.start and self.end == other.end

	def __lt__(self,other):
		if hasattr(self,'start') and hasattr(other, 'start'):
			return self.start < other.start

	def __gt__(self,other):
		if hasattr(self,'start') and hasattr(other, 'start'):
			return self.start > other.start
			

	def __contains__(self,boundary):
		if self.start == boundary:
			return True
		if self.end == boundary:
			return True
		return False


	def set_info(self):
		'''Set start time and end time of the bad_epoch.'''
		if self.ok:
			self.st_sample = int(round(self.start.x))
			self.et_sample = int(round(self.end.x))
			self.duration = self.et_sample - self.st_sample


	def plot(self,plot_annotation = True,plot_correct =True):
		'''plot start / end boundary object add transparent color over time window and plot annotation epoch id.'''
		if not self.visible: return 0
		if self.start_boundary_ok and not self.start.plotted:
			self.start.plot()
		if self.end_boundary_ok and not self.end.plotted:
			self.end.plot()
		if self.ok and not self.plotted:
				plt.axvspan(self.start.x,self.end.x, facecolor = self.color, alpha = 0.1) 
				if plot_annotation:
					self.plot_annotation(plot_correct)


	def set_complete_replot(self):
		'''Set flag to replot epoch (if something changed, e.g. boundary is deleted).'''
		self.plotted = False
		if self.start_boundary_ok: self.start.plotted = False
		if self.end_boundary_ok: self.end.plotted = False


	def in_plot_epoch(self,start,end):
		'''Check whether bad epoch and its boundaries are in the plot window defined by start and end.'''
		self.visible = False
		if self.ok:
			if self.end.x > start and self.end.x < end:
				self.visible = True
			if self.start.x < end and self.start.x > start:
				self.visible = True
			self.start.in_plot_epoch(start,end)
			self.end.in_plot_epoch(start,end)


	def in_bad_epoch(self,x):
		'''Check whether x is within the bad epoch (to check whether mouse position is in this epoch).'''
		if self.ok and x >= self.start.x and x <= self.end.x: return True
		else: return False
		

	def check_boundaries(self):
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
		if self.ok: self.plot()


	def set_start(self,boundary):
		'''Set start boundary.'''
		if boundary.boundary_type == 'start':
			self.start = boundary 
			self.check_boundaries()

	def set_end(self,boundary):
		'''Set end boundary.'''
		if boundary.boundary_type == 'end':
			self.end = boundary
			self.check_boundaries()

	def set_annotation(self,annotation): 
		'''Set the annotation of the bad epoch.'''	
		if type(annotation) == str:
			self.annotation = annotation
		if annotation in self.color_dict.keys(): self.color = self.color_dict[annotation]
		self.plot_annotation()
		self.redraw = True

	def set_correct(self,correct = 'correct'):
		if  self.correct != correct:
			self.redraw = True
		if correct == 'correct':
			self.correct = 'correct'
			# if self.annotation in self.color_dict.keys(): self.color = self.color_dict[self.annotation]
			# else: print('annotation:',self.annotation,'is not in color dict')
		elif correct == 'incorrect':
			self.correct = 'incorrect'
		else: print('unknown input:',correct,'please specify correct or incorrect')

	def add_note(self,note):
		if not hasattr(self,'note'):self.note = ''
		if self.note  == '':
			self.note = note
		else:
			n = self.note.split(',')
			n.append(note)
			self.note = ','.join(n)

	def set_corrector(self,corrector):
		self.corrector = corrector
		

	def plot_annotation(self,plot_correct = True):
		'''Plot the annotation to the plot window.'''
		if self.ok:
			center = (self.st_sample + self.et_sample) / 2
			if self.start.visible:
				plt.annotate(self.annotation,xy=(self.st_sample + 10,1100),color = 'black',fontsize=20)
				plt.annotate(str(self.epoch_id),xy=(self.st_sample + 100, -50),color = 'black')
				plt.annotate(str(self.coder),xy=(self.st_sample + 100, -70),color = 'black')
				if plot_correct:
					if self.correct == 'correct':
						plt.annotate('V', xy=(self.st_sample, -70), color = 'green',fontsize=20)
					if self.correct == 'incorrect':
						plt.annotate('X', xy=(self.st_sample, -70), color = 'red',fontsize=20)
			elif self.end.visible:
				plt.annotate(self.annotation,xy=(self.et_sample - 1000,1100),color = 'black',fontsize =20)
				plt.annotate(str(self.epoch_id),xy=(self.et_sample - 1000,-50),color = 'black')
				plt.annotate(str(self.coder),xy=(self.et_sample + 1000, -70),color = 'black')
				if self.correct == 'correct':
					plt.annotate('V', xy=(self.et_sample - 50, -70), color = 'green',fontsize=20)
				if self.correct == 'incorrect':
					plt.annotate('X', xy=(self.et_sample - 50, -70), color = 'red',fontsize=20)
		

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



class Boundary:
	'''Object that contains information about the start or end of and bad epoch and provides methods to plot and
	check whether it should be plotted.'''

	def __init__(self,x,boundary_type,visible = True):
		''' Create boudnary object with time point and type to specify whether it is an end or start.
		x 				time point
		boundary_type 	str start or end to indicate type of boundary
		'''
		self.x = int(round(x))
		self.boundary_type = boundary_type
		self.visible = visible
		self.plotted = False
		self.plot()

	def __str__(self):
		m = 'Boundary type:\t\t'+ self.boundary_type + '\n'
		m += 'onset in ms: \t\t' + str(self.x) + '\n'
		m += 'visible:\t\t'+str(self.visible) + '\n'
		return m

	def __eq__(self,other):
		if other == None or type(other) != Boundary: return False
		return self.x == other.x and self.boundary_type == other.boundary_type

	def __repr__(self):
		return self.boundary_type + ' ' + str(self.x) 

	def __lt__(self,other):
		if hasattr(other, 'x'):
			return self.x < other.x

	def __gt__(self,other):
		if hasattr(other, 'x'):
			return self.x > other.x

	def in_plot_epoch(self,start,end):
		'''checks whether boundary is in current plot window defined by start and end.'''
		if self.x > start and self.x < end: self.visible = True
		else: self.visible = False


	def plot(self):
		'''Plot a boundary line if it is visible in the current epoch.'''
		if self.visible and not self.plotted:
		
			if self.boundary_type == 'start':
				plt.axvline(self.x,color='tomato',linestyle='-',linewidth=1,alpha=0.5)
			if self.boundary_type == 'end':
				plt.axvline(self.x,color='tomato',linestyle='--',linewidth=1,alpha=0.5)
			self.plotted = True
