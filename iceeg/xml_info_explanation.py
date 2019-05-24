import combine_artifacts
from lxml import etree
import os
import path
import utils
import xml_info

class explanation():
	def __init__(self,name):
		self.name = name

	def __str__(self):
		o = ''
		for k in self.__dict__.keys():
			o += k + ': '+self.__dict__[k] +'\n\n'
		return o

def corpora():
	c = explanation('corpora explanation')
	c.ifadv = '''IFA dialog video corpus, http://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFADVcorpus/
	Reference (IFADV): Van Son, R., Wesseling, W., Sanders, E., & van den Heuvel, H. (2008). The IFADV Corpus: A Free Dialog Video Corpus. In N. Calzolari, K. Choukri, B. Maegaard, J. Mariani, J. Odijk, S. Piperidis & D. Tapias (Eds.), LREC (pp. 501-508). Marrakech: ELRA. Retrieved from http://www.lrec-conf.org/proceedings/lrec2008/pdf/132_paper.pdf (last accessed February 2019).
	'''
	c.cgn = '''corpus gesproken nederlands
	We used version 2 of CGN (Spoken Dutch Corpus)
	https://ivdnt.org/downloads/taalmaterialen/tstc-corpus-gesproken-nederlands
	English description of the corpus
	http://lands.let.ru.nl/cgn/ehome.htm
	Reference (CGN): Oostdijk, N. (2001). The design of the Spoken Dutch Corpus. Language and Computers, 36(1), 105-112.
	'''
	c.sonar = '''The SoNaR corpus
	https://portal.clarin.nl/node/4195
	Reference (SoNaR): Oostdijk, N., Reynaert, M., Hoste, V., Schuurman, I. (2013). The construction of a 500-million-word reference corpus of contemporary written Dutch. In P. Spyns & J. Odijk (Eds.), Essential Speech and Language Technology for Dutch (pp. 219-247). Berlin: Springer.
	'''
	c.cow = '''NLCOW14 corpus, cow: corpora from the web
	https://corporafromtheweb.org/nlcow14/
	Reference: R. Schäfer, “Processing and querying large web corpora with the COW14 architecture,” Proceedings of the 3rd Workshop on Challenges in the Management of Large Corpora (CMLC-3), pp. 28-34. Mannheim: Institut für Sprache, 2015.
	 R. Schäfer and F. Bildhauer, “Building Large Corpora from the Web Using a New Efficient Tool Chain,” Proceedings of the Eight International Conference on Language Resources and Evaluation, Istanbul: ELRA, pp. 486-493, 2012.
	'''
	return c
	
def packages():
	p = explanation('packages explanation')
	p.srilm = '''SRI language modelling toolkit for building and applying statistical language models
	http://www.speech.sri.com/projects/srilm/
	Reference: Stolcke, Andreas (2002): "SRILM - an extensible language modeling toolkit", In ICSLP-2002, 901-904.
	'''
	p.frog = '''frog is a memory-based natural language processing module for Dutch 
	pos (part of speech tag) information is generated with frog
	https://languagemachines.github.io/frog/'
	Reference: Van den Bosch, A., Busser, G.J., Daelemans, W., and Canisius, S. (2007). An efficient memory-based morphosyntactic tagger and parser for Dutch, In F. van Eynde, P. Dirix, I. Schuurman, and V. Vandeghinste (Eds.), Selected Papers of the 17th Computational Linguistics in the Netherlands Meeting, Leuven, Belgium, pp. 99-114
	'''
	p.mne = '''mne: meg and eeg analysis and visualization, (used the python version via pip)
	https://martinos.org/mne/stable/index.html
	Reference: A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, “MEG and EEG data analysis with MNE-Python”, Frontiers in Neuroscience, vol. 7, 2013.
	A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen, “MNE software for processing MEG and EEG data,” NeuroImage, vol. 86, no. 1, pp. 446-460, 2014.
	'''
	p.kaldi = '''Kaldi is a toolkit for speech recognition
	https://kaldi-asr.org/doc/
	https://github.com/kaldi-asr/kaldi
	Reference: Povey, D., Ghoshal, A., Boulianne, G., Burget, L., Glembek, O., Goel, N., ... & Silovsky, J. (2011). The Kaldi speech recognition toolkit (No. CONF). IEEE Signal Processing Society.
	'''
	return p

def block():
	b = explanation('block explanation')
	b.pp_id = 'participant number'
	b.exp_type = 'name identifying the experiment k:news broadcast, o:read-aloud stories, ifadv:spontaneous dialogues'
	b.name = 'name of the block  with participant and experiment info and block number'
	b.experiment_name = 'name of the experiment news-broadcast, read-aloud-stories, spontaneous-dialogues'
	b.corpus = 'corpus the materials were extracted from, see corpora for more information'
	b.block_number = 'the n-th block in the experiment (start at 1), order of blocks was counter balanced across participants'
	b.st = 'start of the block: year-month-day hour:minute:second.subsecond'
	b.et = 'end of the block: year-month-day hour:minute:second.subsecond'
	b.duration = 'length of the block days hour:minute:second.subsecond'
	b.st_sample = 'start sample of the block in the eeg recording'
	b.et_sample = 'end sample of the block in the eeg recording'
	b.duration_sample = 'duration of the block in the eeg recording, sample frequency is 1000 Hz'
	b.sample_inacc = 'the difference in ms between the expected (based on the audio file) and actual eeg recording time' 
	b.nallwords = 'all words in the speech materials'
	b.ncontent_words = 'all content words in the speech materials'
	b.wav_filename = 'the filename of the audio file used in the experiment'
	b.nartefacts = 'number of artefacts in the eeg materials in this block'
	b.fids = '''name of the wav file(s) used in the corpus, 
	read_aloud_stories: wav files are concatenated without pause 
	news_broadcast: there can be a short break between wav files 
	spontaneous_dialogue: always only one audio file'
	start of the current audio file is set in wav_st_sample field in "word"
	'''
	b.fid_st = 'start samples of the fids in the eeg recordings.'
	b.fidset = 'end samples of the fids in the eeg recordings.'
	b.artefact_st = 'start samples of the artefacts in the eeg recordings'
	b.artefacts_et = 'end samples of the artefacts in the eeg recordings'
	b.artefact_fn = 'xml filename with the artefact information'
	b.ica_fn = 'filename,file contains the ica decomposition created with the mne toolkit (see packages)'
	b.eog_fn = 'filename,file contains information about ica and bad channels'
	b.ica_remove_components = 'the indices of the components to be removed (identified as containing eye related activity'
	b.rejected_channels = 'channels that should be removed because they show a lot of artefacts, Fp2 was removed by default'
	b.usability = 'overall quality of the block, mediocre, ok and great were used. Doubtfull and bad were removed from analysis'
	b.vmrk_fn = 'file containing markers with corresponding samples'
	b.vhdr_fn = 'file containing info about channels and their impedance at the start of the recording'
	b.eeg_fn = 'file containing the eeg recordings of this block'
	b.block_duration = 'duration of this block in seconds'
	b.artefacts_duration = 'duration of artefacts in this block'
	return b	
	
def participant():
	p = explanation('participant explanation')
	p.pp_id = 'participant number'
	p.nallwords = 'number of words in the blocks in the experiment (excluding missing blocks)'
	p.ncontent_words='number of content words in the blocks in the exp. (excluding missing blocks)'
	p.nartefact = 'number of artefacts in the block in the exp. (excluding missing blocks)'
	p.dates_sessions = 'dates of the different recording sessions year-month-day'
	p.names_sessions = 'names of the different session (all sessions are always present).'
	p.nblock_missing = 'number of blocks missing from the eeg recordings.'
	p.blocks_duration= 'duration of all blocks present in the eeg recordings'
	p.artefacts_duration= 'duration of all artefacts present in the blocks of eeg recordings'
	return p
	

def session():
	s = explanation('session explanation')
	s.pp_id = 'participant number'
	s.exp_type = 'name identifying the experiment k:news broadcast, o:read-aloud stories, ifadv:spontaneous dialogues'
	s.name = 'name of the session with participant and experiment info'
	s.experiment_name = 'name of the experiment news-broadcast, read-aloud-stories, spontaneous-dialogues'
	s.session_number = 'the n-th session for the participant (1-3), order of sessions was counter balanced across participants'
	s.n_eeg_recordings = '''number of eeg recordings for this session. 
	Due to battery failure a session was sometimes recorded in multiple recordings
	if there are multiple recordings there are multiple eeg, vhdr and vmrk files
	'''
	s.start_exp = 'start of the experiment: year-month-day hour:minute:second.subsecond'
	s.end_exp = 'end of the experiment: year-month-day hour:minute:second.subsecond'
	s.duration = 'length of the experiment days hour:minute:second.subsecond'
	s.nblocks = 'number of block in the experiment'
	s.nallwords = 'all words in the speech materials of this session'
	s.ncontent_words = 'all content words in the speech materials of this session'
	s.nartifacts = 'number of artefacts in the eeg materials in this session'
	s.fids = 'name of wav file for each block sep = , if multiple wav were use in a block sep = |'
	s.fids_missing = 'fids missing because a block is missing from the recording'
	s.usability = 'quality of the block in this session, doubtfull and bad were removed in our analysis'
	s.answer_fn = '''file containing the answers participant gave via button box 
	in response to yes no comprehension questions regarding the speech materials'''
	s.log_fn = '''file containing info of experiment presentation with marker ids and start times and participant info'''
	s.vmrk_fn = 'file containing markers with corresponding samples'
	s.vhdr_fn = 'file containing info about channels and their impedance at the start of the recording'
	s.eeg_fn = 'file containing the eeg recordings of this block'
	s.block_names = 'name of each block in the session, corresponds to the block xml id'
	s.nblock_missing= 'number of blocks missing from eeg recordings (e.g. due to technical issues)'
	s.names_block_missing= 'names of missing blocks'
	s.blocks_duration= 'duration of all blocks present in the eeg recordings'
	s.artefacts_duration= 'duration of all artefacts present in the blocks of eeg recordings'
	return s
	

def word():
	w = explanation('word explanation')
	w.word_utf8_nocode = 'xml_information'
	w.st_sample = 'start sample of word in the eeg recording'
	w.et_sample = 'end sample of word in the eeg recording'
	w.duration_sample = 'duration in samples of the word (sample frequency = 1000)'
	w.st = 'start of word in audio recording in seconds'
	w.et = 'end of word in audio recording in seconds'
	w.eol = 'whether the word has an end of sentence marker'
	w.fid = 'audio filename id'
	w.sid = 'speaker id'
	w.overlap = 'whether the word overlaps with another word'
	w.corpus = 'the corpus the word is extracted from: CGN or IFADV'
	w.register = '''the register of the speech materials, 
	can be spontaneous dialogues, read-aloud stories or news broadcasts
	'''
	w.word = 'word in lower case without diacritics, as used to count in the cow corpus'
	w.fid_st_sample = 'start sample of the wav audio file this word occurs in (st is offset from start audio file)'
	w.block_name ='name of a block in the experiment with participant and experiment info'
	w.word_index_in_block = 'n-th word in the word list (starts at 0)'
	return w

def stats():
	w = explanation('stats explanation')
	w.word_frequency = 'word count in the NLCOW14 corpus'
	w.entropy = '''the entropy over a word predictability distribution (WPD) of approximately
	200,000 words, whereby probability of each word is based on the statistical language model
	based on the NLCOW14 corpus (some language model used for the ppl info.'''
	w.updated_entropy = '''entropy over the WPD, however the probabilities are
	updated with output of an automatic speech recognition (ASR) system Kaldi. We extracted
	a section of audio from word onset to 190 ms and determined the probability of different
	phoneme strings based on this audio section. We matched the phoneme strings with the words
	in the WPD to update the WPD. 
	'''
	w.cross_entropy = '''This measure captures the difference between the probability of a word
	based on preceding words and probability of a word based on the preceding words and auditory 
	input. Computed the cross entropy of the word probability distribution (WPD) based on the
	cow language model and the word probability distribution based on the updated WPD,see updated
	entropy.'''
	w.logprob = '''the log base 10 of the probability of the word based on the cow language model
	this value is identical to the logprob in the ppl (based on the same language model)'''
	w.updated_logprob = '''log base 10 of the probability of the word now updated 
	with ASR system Kaldi. If the updated logprob is less negative than .logprob than the 
	auditory update improved the probability of this word'''
	w.gate = '''duration in milliseconds of the audio section the ASR analysis was computed on.'''
	w.word_number = '''word identifier based on the number of the word in the experiment of pp1.'''
	w.word_code = '''another word identifier, file-id of audio file in corpus (fid), 
	speaker-id (sid) ,chunk_number,word_index_in_sentence,sentence_index_in_block,
	word_number_in_sentence. Fields are seperated with _
	'''
	return w

	
def pos():
	p = explanation('pos explanation')
	p.lemma = '''lemma of the word, 
	pos information is generated with Frog: https://languagemachines.github.io/frog/'''
	p.morphological_segmentation= 'morphological_segmentation of the word'
	p.pos = 'dutch part of speech tag, complete info'
	p.pos_simple = 'dutch part of speech tag'
	p.pos_tag= 'english part of speech tag (based on pos_simple)'
	p.probability_of_tag = 'confidence score of the automatic pos tagger (Frog)'
	p.content_word= 'whether the word is content word'
	p.base_phrase_chunk= 'base phrase chunk of the word'
	return p
	
def ppl():
	p = explanation('ppl explanation')
	p.ngram = '''length of ngram that the word probability is based on
	language models were computed with SRILM'''
	p.oov = 'whether the word is present in the LM vocabulary'
	p.p = '''probability of the word (given the preceding words)
	this probability is based on a language model trained on the NLCOW14 (cow) corpus
	'''
	p.logprob = 'base 10 log of the probability in p (based on cow language model'
	p.p_register = '''probability of the word based on the cow language model 
	interpolated with a language model trained on text with the same register as the
	current experiment, these were always language materials not used in the experiment
	we trained a language model for the different register on the following materials
	spontaneous dialogues: CGN component a (approximately 1 million words)
	news broadcasts: SoNaR auto cues (1 million word subset)
	read-aloud book: SoNaR books (1 million word subset)
	'''
	p.p_other1 = 'probability of word given COW LM interpolated with other (not2) register LM'
	p.logprob_other1 = 'log base 10 of p_other1'
	p.p_other2 = 'probability of word given COW LM interpolated with other (not1) register LM'
	p.logprob_other2 = 'log base 10 of p_other2'
	p.word_id = 'id based on sentence_number, speaker id, word in sentence index and the word itself'
	p.history = 'preceding word the word probability is based on'
	return p

def phoneme_word():
	p = explanation('phoneme_word explanation')
	p.cgn= 'phonemic transcription using the CGN phoneme set'
	p.ipa= 'phonemic transcription using the IPA phoneme set'
	p.nphonemes= 'number of phonemes in the word'
	return p
	
def phoneme():
	p = explanation('phoneme explanation')
	p.phoneme = 'phoneme using cgn phoneme set, ipa version can be found under ipa field'
	p.st_sample = 'start sample of phoneme in the eeg recording'
	p.et_sample = 'end sample of phoneme in the eeg recording'
	p.duration_sample = 'duration of phoneme in the eeg recording (sample frequency is 1000 Hz)'
	return p


def make_explanations():
	o = etree.Element('explanations') 
	names = 'participant,session,block,word,stats,pos,ppl,phoneme_word'
	names += ',phoneme,corpora,packages'
	names = names.split(',')
	for name in names:
		output = globals()[name]().__dict__
		t = etree.SubElement(o,name)
		xml_info.dict2info(output,output.keys(),t)
	return o

def save_xml_explanations():
	o = make_explanations()
	filename = path.data + 'info_explanation.xml'
	with open(filename,'w') as fout:
		fout.write(etree.tostring(o, encoding = 'utf8', pretty_print=True).decode())
	print('saved file to:',filename)

		


