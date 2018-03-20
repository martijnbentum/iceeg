'''The module aggregates timing information for one audio file in the experiment.

block module contains the block class
the module is used by the experiment class
'''
import blinks
import mne
import mne.preprocessing.ica as ica
import numpy as np
import pandas as pd
import path
import preprocessing
import ort
import os
import utils
import windower
import xml_cnn

class block:
	'''The block class aggregates timing information for one audio file in the experiment.'''

	def __init__(self,pp_id,exp_type,vmrk,log,bid, fid2ort = None):
		'''Load general info about start and end time block matches all words 
		to sample numbers in the vmrk marker file.

		Keyword arguments:
		pp_id = participant number int
		exp_type = experiment type ('k' / 'o' / 'ifadv) str
		vmrk = vmrk object, contains marker information 
		log = log object, contains start/end times; name audio file
		bid = block id, int
		fid2ort = dict that maps file id to ort object default = None, if none ort is created
		'''
		self.pp_id = pp_id
		self.exp_type = exp_type
		self.experiment_name = utils.exptype2explanation_dict[self.exp_type]
		self.vmrk = vmrk
		self.log = log
		self.bid = bid
		self.fid2ort = fid2ort
		self.set_info()
		self.load_orts()
		utils.make_attributes_available(self,'ort',self.orts)
		self.eeg_loaded = False
		self.load_blinks()
		self.load_artifacts()
		self.exclude_artifact_words()


	def __str__(self):
		m = '\nBLOCK OBJECT\n'
		m += 'marker\t\t\t'+str(self.marker) + '\n'
		m += 'block missing\t\t'+str(self.block_missing) + '\n'
		m += 'start marker missing\t'+str(self.start_marker_missing) + '\n'
		m += 'end marker missing\t'+str(self.end_marker_missing) + '\n'
		m += 'wav_filename\t\t'+str(self.wav_filename) + '\n'
		m += 'fids\t\t\t'+' - '.join(self.fids) + '\n'
		m += 'sids\t\t\t'+' - '.join(self.sids) + '\n'
		m += 'start sample\t\t'+str(self.st_sample) + '\n'
		m += 'end sample\t\t'+str(self.et_sample) + '\n'
		m += 'duration sample\t\t'+str(self.duration_sample) + '\n'
		m += 'wav duration\t\t'+str(self.wav_dur) + '\n'
		m += 'sample inaccuracy\t' + str(self.sample_inacc) + '\n'
		m += 'start time\t\t'+str(self.st) + '\n'
		m += 'end time\t\t'+str(self.et) + '\n'
		m += 'duration\t\t'+str(self.duration) + '\n'
		m += 'nwords\t\t\t'+str(self.nwords) + '\n'
		m += 'nsentences\t\t'+str(self.nsentences) + '\n'
		m += 'eeg loaded\t\t'+str(self.eeg_loaded) + '\n'
		m += 'nblinks\t\t\t'+str(self.nblinks) + '\n'
		m += 'nartifacts\t\t'+str(self.nartifacts) + '\n'
		m += 'total artifact dur\t'+str(self.total_artifact_duration) + '\n'
		m += 'artifact_perc\t\t'+str(round(self.artifact_perc,3)) + '\n'
		return m

	def __repr__(self):
		return 'Block-object:\t' + str(self.bid) + '\tpp ' + str(self.pp_id) + '\t\texp ' + self.exp_type + ' ' +self.experiment_name + '\t\tnwords: ' + str(self.nwords)


	def set_info(self):
		''''Make block information directly excessable on object and convert time 
		to sample number.
		
		Set information to load ort object which provides time information for
		each word in the audio file
		'''
		self.l = self.log.log
		self.marker = self.l[self.l['block'] == self.bid]['marker'].values[0]
		self.wav_filename = self.log.marker2wav[self.marker]
		self.fids = self.log.marker2fidlist[self.marker]
		self.sids = self.log.fid2sid[self.fids[0]] # sid does not vary in block
		self.wav_dur= self.l[self.l['block'] == self.bid]['wav_duration'].values[0]
		self.set_start_end_times()
		self.st = self.l[self.l['block'] == self.bid]['start_time'].values[0]
		self.et = self.l[self.l['block'] == self.bid]['end_time'].values[0]
		self.duration = self.et - self.st
		if self.exp_type == 'ifadv': 
			self.corpus = 'IFADV'
			self.register = 'spontaneous_dialogue'
		elif self.exp_type == 'o': 
			self.corpus = 'CGN'
			self.register = 'read_aloud_stories'
		elif self.exp_type == 'k': 
			self.corpus = 'CGN'
			self.register = 'news_broadcast' 


	def set_start_end_times(self):
		# print('marker:',self.marker)
		'''Set start and end sample numbers.

		check whether all markers are present otherwise use audio file duration to calculate
			Sample number will point to missing data, needs to be handled elsewhere.
		'''
		self.start_marker_missing = self.marker in self.vmrk.missing_smarkers
		self.end_marker_missing = self.marker + 1 in self.vmrk.missing_emarkers
		self.block_missing = self.start_marker_missing == self.end_marker_missing == True
		if not self.start_marker_missing:
			self.st_sample = self.vmrk.marker2samplen[self.marker]
		if not self.end_marker_missing:
			self.et_sample = self.vmrk.marker2samplen[self.marker+1]
			if not self.start_marker_missing and self.et_sample < self.st_sample:
				self.block_missing = True
		if not self.block_missing:
			if self.start_marker_missing:
				self.st_sample = self.et_sample - self.wav_dur
			elif self.end_marker_missing:
				self.et_sample = self.st_sample + self.wav_dur
			if self.st_sample < 0: self.duration_sample = self.et_sample + self.st_sample
			else: self.duration_sample = self.et_sample - self.st_sample
			self.sample_inacc = abs(self.duration_sample - self.wav_dur)
		if self.block_missing:
			self.st_sample = None
			self.et_sample = None
			self.duration_sample = None
			self.sample_inacc = None


	def load_eeg_data(self, sf = 1000,freq = [0.05,30]):
		'''Load eeg data corresponding to this block.
		sf 		sample frequency, set lower to downsample
		freq 	the frequency of the iir bandpass filter (see preprocessing.py)
		'''
		
		self.raw = preprocessing.load_block(self, sf = sf,freq = freq)
		if self.raw != 0: 
			self.eeg_loaded = True
		else:
			print('could not load raw')

	def unload_eeg_data(self):
		'''Unload eeg data.'''
		if hasattr(self,'raw'):
			delattr(self,'raw')
		self.eeg_loaded = False


	def load_blinks(self, offset = 1500):
		'''Load blink sample number as found with automatically classified blink model.'''
		try:
			self.blinks_text= open(path.blinks + windower.make_name(self)+ '_blink-model.classification').read()
			self.blink_peak_sample = np.array([int(line.split('\t')[2]) for line in self.blinks_text.split('\n')])
			self.nblinks = len(self.blink_peak_sample)
			self.blink_start = self.blink_peak_sample - offset
			self.blink_end = self.blink_peak_sample + offset
		except:
			print('could not load blinks')
			self.blinks_text,self.blink_peak_sample,self.nblinks = 'NA','NA','NA'
			self.blink_start, self.blink_end = 'NA','NA'


	def load_artifacts(self):
		'''Loads automatically generated artifact annotations.
		WIP: extend to be able to specify which annotation type to load (manual / automatic)
		'''
		try:
			self.xml = xml_cnn.xml_cnn(self)
			self.xml.load()
			self.xml.xml2bad_epochs()
			self.artifacts = [a for a in self.xml.artifacts if a.annotation == 'artifact']
			self.nartifacts = len(self.artifacts)
			self.start_artifacts = [a.st_sample/1000 for a in self.artifacts]
			self.duration_artifacts = [a.duration/1000 for a in self.artifacts]
			self.total_artifact_duration = sum(self.duration_artifacts)
			self.artifact_perc = self.total_artifact_duration/(self.duration_sample/1000)
		except:
			print('could not load xml cnn, artefacts.')
			self.artifacts,self.start_artifacts,self.duration_artifacts = 'NA', 'NA','NA'
			self.total_duration,self.total_artifact_duration,self.artifact_perc, self.nartifacts = 0,0,0,0


	def fit_ica(self):
		'''Fit ica on block object.
		EEG data is bandpassed filtered on 1-30 Hz, the ica solution can be used on data
		without or different bandpass filter see:
		Winkler et al.
		On the influence of high-pass filtering on ICA-based artifact reduction in EEG-ERP
		WIP: extend to be able to set ica approach (with or without artifact rejection, filename)
		'''
		if not self.eeg_loaded: 
			try:self.load_eeg_data(freq = [1,30])
			except: return 0
		self.ica = ica.ICA()
		self.ica_e = ica.ICA()
		if self.artifacts != 'NA':
			self.raw.annotations = mne.Annotations(self.start_artifacts,self.duration_artifacts,'BAD')
		self.ica.fit(self.raw,reject_by_annotation = False)
		self.eog_comp, self.eog_scores = self.ica.find_bads_eog(self.raw,reject_by_annotation = False)
		self.ica.save(path.ica_solutions + windower.make_name(self) + '_all-data-ica.fif')

		if self.artifacts != 'NA':
			self.ica_e.fit(self.raw,reject_by_annotation = True)
			self.eog_comp_e, self.eog_scores_e = self.ica_e.find_bads_eog(self.raw,reject_by_annotation = False)
			self.ica_e.save(path.ica_solutions + windower.make_name(self) + '_no-artifact-ica.fif')
		else: self.eog_comp_e,self.eog_scores_e = self.eog_comp, self.eog_scores
		
		self.write_eog()


	def write_eog(self):
		'''Write text file with correlation between eog channels and ic from ica solution.
		performs ica with and without artifact rejection
		WIP: assumes that both all_data and artifact rejection is performed, 
		should be more general.
		'''
		if self.artifacts == 'NA': xml_ok = 'xml_absent'
		else: xml_ok = 'xml_present'
		eog = ','.join(map(str,self.eog_comp))
		veog_scores = ','.join([str(self.eog_scores[0][i]) for i in self.eog_comp])
		heog_scores = ','.join([str(self.eog_scores[1][i]) for i in self.eog_comp])

		eog_e = ','.join(map(str,self.eog_comp_e))
		veog_scores_e = ','.join([str(self.eog_scores_e[0][i]) for i in self.eog_comp_e])
		heog_scores_e = ','.join([str(self.eog_scores_e[1][i]) for i in self.eog_comp_e])

		output = '\t'.join([eog,veog_scores,heog_scores,eog_e,veog_scores_e,heog_scores_e,xml_ok,str(self.artifact_perc)])

		fout = open(path.ica_solutions + windower.make_name(self) + '_eog.txt','w')
		fout.write(output)
		fout.close()

	def exclude_artifact_words(self):
		'''Set words which overlap with artifact timeframe to usable == False, recounts nwords.
		sets the artifact id to the word object 
		WIP: maybe seperate field for artifact usability on the word object
		'''
		if self.artifacts == 'NA' or type(self.st_sample) != int: return 0
		for w in self.words:
			for a in self.artifacts:
				o = utils.compute_overlap(w.st_sample - self.st_sample,w.et_sample - self.st_sample,a.st_sample,a.et_sample)
				if o > 0:
					w.set_artifact(a)
					break
		self.nallwords = self.nwords
		self.nwords = len([w for w in self.words if w.usable])
		self.nexcluded = self.nallwords - self.nwords

	def load_orts(self):
		'''create the ort object for all fids (file id) in the block  

		A block consists of 1 experimental audio file which can consist of 
		multiple wav from comp-o or comp-k (always 1 for ifadv)
		Files from comp-k are concatenated with an offset of 900 ms

		The ort object is created based on corpus transcriptions of CGN and IFADV
		For each word in the transcription the time is matched to a sample number
		by calculating the offset of the word from the marker in the vmrk object 
		which signals the onset of the experimental audio file
		'''
		k_wav_offset_second = 0.9
		k_wav_offset_sample= 900
		self.orts = []
		self.words = []
		self.sentences = []
		total_fids_offset = 0
		self.ncontent_words = 0
		for i,fid in enumerate(self.fids):
			# create ort object for fid
			if self.fid2ort and fid in self.fid2ort.keys():
				# orts are already made load them from dictonary
				self.orts.append(self.fid2ort[fid])
			else:
				# otherwise make them now
				self.orts.append(ort.Ort(fid,self.sids[0],corpus = self.corpus,register = self.register))
				if self.exp_type == 'ifadv':
					# add extra speaker for ifadv and check for overlap between speakers
					self.orts[-1].add_speaker(self.sids[1])
					self.orts[-1].check_overlap()
			for w in self.orts[-1].words:
				# set sample offset (when did the recording start playing 
				# in sample time, if wav consist of multiple wav add offset
				# k (news) also had an pause between files of .9 seconds
				if self.exp_type == 'k' and i > 0:
					total_fids_offset += last_fid_duration + k_wav_offset_sample
				elif self.exp_type == 'o' and i > 0:
					total_fids_offset += last_fid_duration 
				else:
					total_fids_offset = 0
				# set samplenumber for each word
				if not self.block_missing:
					w.set_times_as_samples(offset=total_fids_offset + self.st_sample)
			if self.exp_type in ['o','k']:
				# get the duration of the las wav file
				last_fid_duration = self.log.fid2dur[fid]
			# add all words and sentences of the ort object corresponding to fid to the block
			self.words.extend(self.orts[-1].words)
			self.sentences.extend(self.orts[-1].sentences)
			for speaker in self.orts[-1].speakers:
				self.ncontent_words += speaker.ncontent_words
		# count number of words in this block
		self.nwords = len(self.words)
		self.nsentences= len(self.sentences)
		self.norts = len(self.orts)

	def block2name(self):
		return 'pp'+str(self.pp_id) + '_exp-' + str(self.exp_type) + '_bid-' + str(self.bid)



			
