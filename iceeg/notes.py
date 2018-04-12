'''The note object contains information about an annotated eeg block.
during annotation notes should be edited in a gui environment.
The gui environment is called with .edit() on the note object.
Notes are stored in xml format with the xml_notes module.
'''

import path
from matplotlib import pyplot as plt
import os
import time
import windower
import wx
import xml_notes

class note():
	'''Data object with annotation information provided by annotator.'''
	def __init__(self,name,coder = 'martijn',note_type = 'general',annotation_type = 'artifact',note = '',ch_notes = [],general_notes = [],try_load = True,created = '',edited='',last_edited= '',old_logs = []):
		'''Create new note object or load a previously created note based on name.
		(block name, windower.make_name)
		Notes should be edited in the gui environment which can be called with .edit() on the note

		name 		block name based on windower.make_name(block)
		coder 		annotator name
		note_type 	type of annotation
		annotat... 	type of annotation
		note 		str, note to be set on note object
		ch_notes 	channel notes
		general... 	general information
		try_load 	wheter to try to load a previously created noted corresponding with name
		created 	date time creation of note
		last_edi... last time an edit was made
		old_logs 	previously made note entries, called logs in xml_notes
		'''
		self.name = name
		self.filename = path.notes + 'note_'+self.name + '.xml'
		self.coder = coder
		self.note_type = note_type
		self.annotation_type = annotation_type
		self.ch = [ch for ch in open(path.data + 'channel_names_selection.txt').read().split('\n') if ch]
		self.ch_annot = ['na'] * len(self.ch)
		if created == '':self.created= time.strftime('%d %B %Y   %H:%M:%S')
		else: self.created = created
		self.edited = False
		self.note,self.general_notes,self.old_logs = '',{},{}
		self.clear_channel_notes()
		if try_load and os.path.isfile(self.filename): self.load()
		else: self.set_values(note,ch_notes,general_notes,old_logs)
		

	def load(self):
		'''Load a previously created note from xml file corresponding with name.'''
		xn = xml_notes.xml_notes(filename = self.filename)
		xn.load_xml()
		xn.xml2note()
		self.__dict__.update(xn.note.__dict__)


	def save(self):
		'''Save current note to xml.'''
		xn = xml_notes.xml_notes(note = self)
		xn.note2xml()
		xn.write()


	def __str__(self):
		m = 'name\t\t\t'+self.name + '\n'
		m += 'coder\t\t\t'+self.coder+'\n'
		m += 'note_type\t\t'+self.note_type+'\n'
		m += 'annotation_type\t\t'+self.annotation_type+'\n'
		m += 'created\t\t\t'+self.created+'\n'
		m += 'edited\t\t\t'+self.last_edited+'\n'
		return m

	def __repr__(self):
		return 'note-object ' + self.coder + ' '+self.name +'\tedited: '+self.last_edited 

	def str(self):
		'''Create str containing note information'''
		m = 'Info\n----\n'
		m += self.__str__()
		m += '\n'
		if self.note != '':
			m += 'Note\n----\n'
			m += self.note
			m += '\n\n'
		else: m+= 'No Notes available\n\n'
		if [line for line in dict_to_list(self.ch_notes) if line[1] != 'na'] == []:
			m+= 'No Channel Notes available'
		else:
			m += 'Channel notes\n'+'-'*13+'\n'
			m += '\n'.join(['\t'.join(line) for line in dict_to_list(self.ch_notes) if line[1] != 'na'])
		m += '\n'
		m += '\nGeneral notes\n'+'-'*13+'\n'
		m += '\n'.join([line[0].ljust(25) + line[1] for line in dict_to_list(self.general_notes)])
		return m


	def show(self):
		'''Show info about the note.'''
		print(self.str())


	def set_note(self,note = ''):
		'''Set note of the note object, str containing information about block /annotation.'''
		if type(note) != str: pass
		elif self.note != note and note == '': print('use clear_note() to delete note')
		else:
			self.note = note
			self.edited = True
	

	def clear_note(self):
		'''Set note to empty string.'''
		self.note = ''
		self.edited = True
		self.set_last_edited()


	def set_channel_notes(self,ch_notes = {}):
		'''Set the notes for each channel.'''
		if type(ch_notes) == list:
			try: ch_notes = dict(ch_notes)
			except: print('could not convert:',ch_notes,'to dict')
		elif ch_notes != self.ch_notes and ch_notes == {}:
			print('use clear channel_notes() to clear annotations.')
		if type(ch_notes) == dict: 
			for ch in ch_notes.keys():
				self.ch_notes[ch] = ch_notes[ch]
			self.edited = True
		else: print('could not set:',ch_notes,type(ch_notes)) 


	def clear_channel_notes(self):
		'''Set all channel notes to 'na' (default).'''
		self.ch_notes = {}
		for i in range(len(self.ch)):
			self.ch_notes[self.ch[i]] = self.ch_annot[i] 
		self.edited = True
		self.set_last_edited()
		

	def set_general_notes(self,general_notes = {}):
		'''Set general notes, contains general information about block / annotation.'''
		if type(general_notes) == list:
			try:
				general_notes= dict(general_notes)
				for key in general_notes.keys(): 
					self.general_notes[key] = general_notes[key]
				self.edited = True
			except: print('could not convert:',general_notes,'to dict')
		elif type(general_notes) == dict:
				self.general_notes = general_notes
				self.edited = True
		else: print('could not set:',general_notes,type(general_notes))


	def set_old_logs(self,old_logs = {}):
		'''Set the previous note to the old logs.
		each time a note is edited the current note is added to the log
		'''
		if type(old_logs) == list:
			try:
				old_logs= dict(old_logs)
				for key in old_logs.keys(): 
					self.old_logs[key] = old_logs[key]
				self.edited = True
			except: print('could not convert:',old_logs,'to dict')
		elif type(old_logs) == dict:
				self.old_logs = old_logs
				self.edited = True
		else: print('could not set:',old_logs,type(old_logs))

	def set_values(self,note = '',ch_notes = {}, general_notes = {},old_logs= {}):
		'''Set note channel notes and general notes and old logs.''' 
		self.edited = False
		if note == '':pass
		else: self.set_note(note)
		if ch_notes== {}:pass
		else: self.set_channel_notes(ch_notes)
		if general_notes== {}:pass
		else: self.set_general_notes(general_notes)
		if old_logs== {}:pass
		else: self.set_old_logs(old_logs)
		self.set_last_edited()
			

	def set_last_edited(self):
		'''Set the time the note was last edited.'''
		if self.edited == True:
			self.last_edited = time.strftime('%d %B %Y   %H:%M:%S')
			self.edited = False

	def edit(self):
		'''Create a wxpython interface to edit the note.'''
		plt.ion()
		plt.plot((1,2,3))
		ex = wx.App()
		plt.close()
		note_gui(self,None)
		ex.MainLoop()	 


class note_gui(wx.Frame):
	'''Create a wxpython interface to edit the note and show the current state of the note.'''
	def __init__(self,note, *args, **kw):
		super(note_gui, self).__init__(*args, **kw) 
		self.ch = [ch for ch in open(path.data + 'channel_names_selection.txt').read().split('\n') if ch]
		self.note = note
		self.note.show()
		print(self.note)
		self.make_screen()
	

	def make_screen(self):   
		'''Create the gui environment.'''
		pnl = wx.Panel(self)
		pnl.Bind(wx.EVT_SET_FOCUS,self.onFocus)
		pnl.Bind(wx.EVT_KILL_FOCUS,self.offFocus)
		pnl.Bind(wx.EVT_KEY_DOWN, self.handle_key)
		btn = wx.Button(pnl, label='Ok', pos=(800, 550), size=(60, -1))
		btn.Bind(wx.EVT_BUTTON, self.OnClose)

		usability= ['na','great', 'ok', 'mediocre', 'doubtfull', 'bad']
		gq= ['na','none', 'some', 'present', 'strong', 'extreme']
		annot= ['na','easy', 'ok', 'doubts', 'difficult']
		options= ['na','perfect', 'good','ok', 'shaky', 'bad']

		x1 = 80
		x2 = 350

		self.general_labels = 'usability,alpha,drift,noise,annot_diff,artifact,channel,blink'.split(',')
		self.general_choices = [usability,gq,gq,gq,annot,options,options,options]
		self.general_xpos = [x1,x1,x1,x1,x1,x2,x2,x2]
		self.general_ypos = [35,75,115,155,195,75,115,155]
		# self.general_id = [self.general_choices.index(gc) for gc in self.general_choices]

		self.general_cb = []

		for i,label in enumerate(self.general_labels):
			self.general_cb.append(wx.ComboBox(pnl, pos=(self.general_xpos[i], self.general_ypos[i]), choices=self.general_choices[i], style=wx.CB_READONLY,id =i))
			self.general_cb[-1].Bind(wx.EVT_COMBOBOX, self.general_select)
			if self.note.general_notes != {}:
				self.general_cb[-1].SetValue(self.note.general_notes[label])
			wx.StaticText(pnl, label=self.general_labels[i], pos=(self.general_xpos[i]-70, self.general_ypos[i]))


		g= wx.StaticText(pnl, label='general', pos=(5, 3))
		header= wx.Font(20,wx.DEFAULT,wx.NORMAL,wx.BOLD)
		g.SetFont(header)
		aq= wx.StaticText(pnl, label='automatic annotation quality', pos=(220, 3))
		aq.SetFont(header)
		c= wx.StaticText(pnl, label='channel notes', pos=(600, 3))
		c.SetFont(header)

		channel= ['na','noise', 'drift', 'garbage', 'jumps','?annot']
		self.ch_comb = []
		self.ch_labels= []
		ypos = 35
		xpos = 600
		second_column = False
		for i,ch in enumerate(self.ch):
			self.ch_comb.append( wx.ComboBox(pnl, pos=(xpos, ypos), choices=channel, 
				style=wx.CB_READONLY,id= self.ch.index(ch)))
			self.ch_labels.append( wx.StaticText(pnl, label=ch, pos=(xpos + 100, ypos+5)))
			self.ch_comb[-1].Bind(wx.EVT_COMBOBOX, self.channel_select)
			self.ch_comb[-1].SetValue(self.note.ch_notes[ch])
			if i >= len(self.ch)/2 and not second_column: 
				xpos = 800
				ypos = 5
				second_column = True
			ypos += 30
		
		n =wx.StaticText(pnl, pos=(5, 270), label= 'notes')
		n.SetFont(header)
		self.t =wx.TextCtrl(pnl, pos=(5, 300), size=(450, 300),style=wx.TE_MULTILINE,value = self.note.note)
		pnl.Bind(wx.EVT_KEY_DOWN, self.handle_key)
		self.t.Bind(wx.EVT_KEY_UP, self.handle_key)
		self.SetSize((1000, 650))
		self.SetTitle('martijn')
		self.Centre()
		g.SetFocus()
		self.Show(True)			 
		self.pnl = pnl
	

	def onFocus(self,e):
		'''Debugging function.'''
		print('in focus')

	def offFocus(self,e):
		'''Debugging function.'''
		print('out of focus')

	def channel_select(self, e):
		'''Debugging function.'''
		i = e.GetString()
		print(i,self.ch[e.Id])

	def general_select(self, e):
		'''Debugging function.'''
		i = e.GetString()
		print(i,self.general_labels[e.Id])
		

	def OnClose(self, e, save = True):
		'''Save the note to xml when the gui environment is closed.'''
		channel_list,general_list = [], []
		for cb in self.ch_comb:
			channel_list.append([self.ch[cb.Id],cb.GetValue()])
		for cb in self.general_cb:
			general_list.append([self.general_labels[cb.Id],cb.GetValue()])
		note = self.t.GetValue()
		self.note.set_values(note=note,ch_notes=channel_list,general_notes= general_list)
		self.Close(True)	
		if save:
			self.note.save()
	

	def handle_key(self, e):
		'''Close the gui enviroment (and save) on ctrl press.'''
		kc = e.GetKeyCode()
		if kc == 396: self.OnClose(e)
			   

def dict_to_list(d):
	'''Create a list of lists from a dict.'''
	return list(zip(d.keys(),d.values()))
