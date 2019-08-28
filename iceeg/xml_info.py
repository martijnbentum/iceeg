import combine_artifacts
import experiment as e
from lxml import etree
import os
import path
import utils
import word2ce

wtd = utils.load_dict_wordtype2freq() 

def make_all_info(start = 1):
	for i in range(start,49):
		print('making info for participant:',i)
		make_info(i)
	print('all done')

def make_info(p):
	p = check_p(p)
	d = path.xml_info + 'PP' + str(p.pp_id) + '/'
	dw = d + 'WORDS/'
	if not os.path.isdir(d): os.mkdir(d)
	if not os.path.isdir(dw): os.mkdir(dw)
	o = participant2xml(p)
	save(o,d + 'participant.xml')
	o = etree.Element('sessions')
	for s in p.sessions:
		session2xml(s,o)
	save(o,d + 'sessions.xml')
	o = etree.Element('blocks')
	for b in p.blocks:
		if b.block_missing: continue
		block2xml(b,o)
		ow = block2wordsxml(b)
		save(ow,dw + b.name + '.xml')
	save(o,d + 'blocks.xml')

def save(xml,name):
	t = etree.tostring(xml, encoding = 'utf8',pretty_print=True)
	with open(name,'wb') as fout:
		fout.write(t)
	print('saved:',name)
	
	

def participant2xml(p,goal = None):
	p = check_p(p)
	if goal == None: o = etree.Element('participant',id = str(p.pp_id))
	else: o = etree.SubElement(goal,'participant') # NO ID HERE??
	pp_id = p.pp_id
	nallwords = sum([b.nallwords for b in p.blocks if not b.block_missing and type(b.nallwords) == int])
	ncontent_words= sum([b.ncontent_words for b in p.blocks if not b.block_missing])
	nartefacts = sum([b.nartifacts for b in p.blocks if not b.block_missing])
	dates_sessions = ','.join([str(s.start_exp).split(' ')[0] for s in p.sessions])
	names_sessions = ','.join([s.name for s in p.sessions])
	names_block_missing = ','.join([b.name for b in p.blocks if b.block_missing])
	nblocks_missing =  len([b for b in p.blocks if b.block_missing])
	blocks_duration = sum([b.duration_sample/1000 for b in p.blocks if not b.block_missing])
	artefacts_duration = sum([b.total_artifact_duration for b in p.blocks if not b.block_missing])
	names = 'pp_id,nallwords,ncontent_words,nartefacts,dates_sessions,names_sessions'
	names += ',nblocks_missing,names_block_missing,blocks_duration,artefacts_duration'
	names = names.split(',')
	values = [pp_id,nallwords,ncontent_words,nartefacts,dates_sessions,names_sessions]
	values += [nblocks_missing,names_block_missing,blocks_duration,artefacts_duration]
	d = make_dict(names,values)
	return dict2info(d,names,o)
	


def block2wordsxml(b):
	o = etree.Element('words', id = b.name + '-words')
	for i, word in enumerate(b.words):
		o = word2xml(word,i,o,b.name)
	return o
		

def word2xml(w,index,goal,name):
	o = etree.SubElement(goal,'word',id =name +'-word-index-'+str(index))
	names ='word_utf8_nocode,st_sample,et_sample,duration_sample,st,et,eol,fid,sid'
	names += ',overlap,corpus,register'
	names = names.split(',')
	o = dict2info(w.__dict__,names,o)
	wav_st_sample = str(w.sample_offset)
	block_name = name
	word_index_in_block = index
	word = w.word_utf8_nocode_nodia().lower()
	names = 'word,fid_st_sample,block_name,word_index_in_block'
	names = names.split(',')
	values = [word,wav_st_sample,block_name,word_index_in_block]
	d = make_dict(names,values)
	o = dict2info(d,names,o)
	word2stats2xml(w,o)
	if hasattr(w,'pos'): pos2xml(w.pos,o)
	if hasattr(w,'ppl'): ppl2xml(w.ppl,o)
	if w.awd_ok:
		w.add_phoneme_word()
		if hasattr(w,'phoneme_word'):phoneme_word2xml(w.phoneme_word,o)
	return goal

def phoneme_word2xml(pw,goal):
	ipd = utils.load_ipa_dict()
	o = etree.SubElement(goal,'phoneme_word')
	cgn = pw.phon_word
	ipa = ''.join([ipd[p.phoneme] for p in pw.phonemes])
	nphonemes = pw.nphonemes
	names = ['cgn','ipa','nphonemes']
	values = [cgn,ipa,nphonemes]
	d = make_dict(names,values)
	o = dict2info(d,names,o)
	for i,p in enumerate(pw.phonemes):
		phoneme2xml(p,i,o)
	return goal

def phoneme2xml(p,index,goal):
	ipd = utils.load_ipa_dict()
	o = etree.SubElement(goal,'phoneme')
	index = str(index)
	cgn = p.phoneme
	ipa = ipd[p.phoneme]
	st_sample = p.st_sample
	et_sample = p.et_sample
	duration_sample = p.duration_sample
	names = 'index,cgn,ipa,st_sample,et_sample,duration_sample'.split(',')
	values = [index,cgn,ipa,st_sample,et_sample,duration_sample]
	d = make_dict(names,values)
	return dict2info(d,names,o)
	

def word2stats2xml(w,goal):
	o = etree.SubElement(goal,'stats')
	word = w.word_utf8_nocode_nodia().lower()
	if word in wtd.keys(): word_frequency = wtd[word]
	else: word_frequency = 'NA'
	gate = 190
	word_code,word_number,output = word2ce.word2info(w,gate)
	entropy,ud_entropy,cross_entropy,logprob,ud_logprob= output
	names = 'word_frequency,entropy,updated_entropy,cross_entropy,logprob,updated_logprob,gate'
	names += ',word_number,word_code'
	names = names.split(',')
	values = [word_frequency,entropy,ud_entropy,cross_entropy,logprob,ud_logprob,gate]
	values += [word_number,word_code]
	d = make_dict(names,values)
	return dict2info(d,names,o)

def pos2xml(pos,goal):
	o = etree.SubElement(goal,'pos')
	pos.set_info()
	names = 'lemma,morphological_segmentation,pos,pos_simple,pos_tag,probability_of_tag'
	names += ',content_word,base_phrase_chunk'
	names = names.split(',')
	return dict2info(pos.__dict__,names,o)

def ppl2xml(ppl,goal):
	o = etree.SubElement(goal,'ppl')
	names = 'ngram,oov,p,logprob,p_register,logprob_register,p_other1'
	names += ',logprob_other1,p_other2,logprob_other2'
	names = names.split(',')
	o = dict2info(ppl.__dict__,names,o)
	word_id = ppl.word_id
	wis = int(word_id.split('_')[-2])
	ngram = ppl.ngram
	if wis > (ngram-1): start = wis - (ngram-1)
	else: start = 0
	history = ','.join(ppl.sentence.split(' ')[start:wis])
	names = 'word_id,word_index_sentence,history'.split(',')
	values = [word_id,wis,history]
	d = make_dict(names,values)
	return dict2info(d,names,o)


def block_name2info(name):
	pp_id=str(utils.name2pp_id(name))
	exp_type=utils.name2exp_type(name)
	bid=str(utils.name2bid(name))
	return pp_id, exp_type, bid


def block2xml(b,goal = None):
	if goal == None:o = etree.Element('block', id = b.name)
	else:o = etree.SubElement(goal,'block',id = b.name)
	names ='pp_id,exp_type,name,experiment_name,corpus,block_number'
	names += ',st,et,duration,st_sample,et_sample,duration_sample,sample_inacc'
	names += ',nallwords,ncontent_words,wav_filename'
	names = names.split(',')
	o = dict2info(b.__dict__,names,o)
	nartefacts = str(b.nartifacts)
	fids = ','.join(b.fids)
	fid_st,fid_et = get_fid_start_end(b)
	st, et = get_artefact_start_end(b)
	ica_fn, eog_fn,ica_remove_components,rejected_channels,artefact_fn,usability = get_block_info(b)
	if type(b.vmrk.vmrk_fn) == str: vmrk_fn = b.vmrk.vmrk_fn
	elif b.marker in b.vmrk.marker2vmrk_fn.keys(): vmrk_fn = b.vmrk.marker2vmrk_fn[b.marker]
	elif len(list(set(b.vmrk.marker2vmrk_fn.values()))) == 1:vmrk_fn = list(b.vmrk.marker2vmrk_fn.values())[0]
	else: vmrk_fn = b.vmrk.vmrk_fn
	if type(vmrk_fn) == list:vmrk_fn, vhdr_fn, eeg_fn = vmrk2fn(b.vmrk)
	else:
		vhdr_fn, eeg_fn = get_vhdr_eeg_f(vmrk_fn)
		vmrk_fn = fix_fn(vmrk_fn)
	block_duration = str(b.duration_sample/1000) if not b.block_missing else 'NA'
	artefacts_duration = b.total_artifact_duration if not b.block_missing else 'NA'
	names = 'nartefacts,fids,fid_st,fid_et,artefact_st,artefact_et,artefact_fn,ica_fn,eog_fn'
	names += ',ica_remove_components,rejected_channels,usability,vmrk_fn,vhdr_fn'
	names += ',eeg_fn,block_duration,artefacts_duration'
	names = names.split(',')
	values = [nartefacts,fids,fid_st,fid_et,st,et,artefact_fn,ica_fn,eog_fn,ica_remove_components]
	values += [rejected_channels,usability,vmrk_fn,vhdr_fn,eeg_fn]
	values += [block_duration,artefacts_duration]
	d = make_dict(names,values)
	return dict2info(d,names,o)

def get_fid_start_end(b):
	st, et = [], []
	last_fid = ''
	for w in b.words:
		if w.fid != last_fid:
			st.append(str(w.sample_offset))
			if b.exp_type == 'ifadv':
				et.append(str(w.sample_offset + 900000))
			else: et.append(str(w.sample_offset + b.log.fid2dur[w.fid]))
		last_fid = w.fid
	return ','.join(st), ','.join(et)

def get_artefact_start_end(b):
	st, et = [], []
	if not hasattr(b,'artifacts') or b.artifacts == 'NA': return 'NA', 'NA'
	for a in b.artifacts:
		st.append(str(b.st_sample + a.st_sample))
		et.append(str(b.st_sample + a.et_sample))
	return ','.join(st), ','.join(et)
	

def get_block_info(b):
	ica_fn,eog_fn,ica_remove_components,rejected_channels = ['NA']*4
	try:b.load_ica() 
	except: print('could not load ica')
	if hasattr(b,'eog'):
		ica_fn = fix_fn(b.ica_filename)
		eog_fn = fix_fn(path.ica_solutions+b.eog.filename)
		ica_remove_components = ','.join(map(str,b.eog.comps))
		rejected_channels = ','.join(b.eog.rejected_channels)
	artefact_fn = combine_artifacts.make_xml_name(b.name)
	if not os.path.isfile(artefact_fn): artefact_fn = 'NA'
	else: artefact_fn = fix_fn(artefact_fn)
	if hasattr(b,'xml'): usability = b.xml.usability
	else: usability = 'NA'
	return ica_fn, eog_fn,ica_remove_components,rejected_channels,artefact_fn, usability
	

def session2xml(s,goal = None):
	if goal == None:o = etree.Element('session', id = s.name)
	else:o = etree.SubElement(goal,'session', id= s.name)
	names ='pp_id,exp_type,name,experiment_name,session_number'
	names += ',n_eeg_recordings,start_exp,end_exp,duration,nblocks'
	names += ',nallwords,ncontent_words,nartifacts'
	names = names.split(',')
	o = dict2info(s.__dict__,names,o)
	answer_fn, log_fn, vmrk_fn, vhdr_fn, eeg_fn = make_fn(s)
	block_names = ','.join([b.name for b in s.blocks if not b.block_missing])
	fids = ','.join(['|'.join(b.fids) for b in s.blocks if not b.block_missing])
	fids_missing = ','.join(['|'.join(b.fids) for b in s.blocks if b.block_missing])
	usability = ','.join(s.__dict__['usability'])
	nblocks_missing =  len([b for b in s.blocks if b.block_missing])
	names_block_missing = ','.join([b.name for b in s.blocks if b.block_missing])
	blocks_duration = sum([b.duration_sample/1000 for b in s.blocks if not b.block_missing])
	artefacts_duration = sum([b.total_artifact_duration for b in s.blocks if not b.block_missing])
	names = 'fids,fids_missing,usability,answer_fn,log_fn,vmrk_fn,vhdr_fn,eeg_fn'
	names += ',block_names,nblocks_missing,names_block_missing,blocks_duration,artefacts_duration'
	names = names.split(',')
	values = [fids,fids_missing,usability,answer_fn,log_fn,vmrk_fn,vhdr_fn,eeg_fn]
	values += [block_names,nblocks_missing,names_block_missing,blocks_duration,artefacts_duration]
	d = make_dict(names,values)
	return dict2info(d,names,o)

def usable2xml(usable):
	'''add usability type counts for the labels in the usable list.
	made for participant xml to give an overview of the quality of the recordings
	'''
	o = etree.Element('usability')
	d = count_usability(usability)
	names = 'great,ok,medicre,doubtfull,bad'.split(',')
	return dict2info(d,names,o)

def make_fn(s):
	'''Create relevant filenames for a session object.'''
	answer_fn = fix_fn(s.log.answer_fn)
	log_fn = fix_fn(s.log.log_fn)
	vmrk_fn, vhdr_fn, eeg_fn = vmrk2fn(s.vmrk)
	return answer_fn, log_fn, vmrk_fn, vhdr_fn, eeg_fn

def vmrk2fn(vmrk):
	'''extract eeg filenames from a vmrk object.'''
	v = vmrk.vmrk_fn
	vmrk_fn = fix_fn(v) if type(v) == str else ','.join([fix_fn(f) for f in v])
	vhdr_fn, eeg_fn = get_vhdr_eeg_fn(v)
	return vmrk_fn, vhdr_fn, eeg_fn


def fix_fn(fn):
	'''Extract the relevant path from the complete path (removes the idiosyncratic start).'''
	t = fn.split('/')
	for i,e in enumerate(t):
		if e in ['EEG','EEG_DATA_ifadv_cgn']: return '/'.join(t[i:])
	print('could not find default start directory, returning whole filename')
	return fn
		
def get_vhdr_eeg_fn(vmrk):
	'''make vhdr and eeg files from vmrk file and check they are present.'''
	if type(vmrk) == list:
		temp = [get_vhdr_eeg_f(v) for v in vmrk]
		vhdr = [l[0] for l in temp]
		eeg = [l[1] for l in temp]
		return ','.join(vhdr), ','.join(eeg)
	else: return get_vhdr_eeg_f(vmrk) 

def get_vhdr_eeg_f(vmrk):
	o = []
	for ex in ['.vhdr','.eeg']:
		t = vmrk.replace('.vmrk',ex)
		if not os.path.isfile(t):
			raise ValueError(t,'could not find',ex,'file')
		o.append(fix_fn(t))
	return o

def dict2info(d,names,goal):
	'''create xml based on a dictionary.'''
	for name in names:
		e = etree.SubElement(goal,name)
		e.text = str(d[name])
	return goal
		
def make_dict(names,values):
	'''create a dictionary based on list of names and values.'''
	d = {}
	for i,name in enumerate(names):
		d[name] = values[i]
	return d
	
		
def count_usability(usable):
	'''count the number of usablility types in a list of usability labels.'''
	o = {}
	for label in 'great,ok,medicre,doubtfull,bad'.split(','):
		o[label] = usable.count(label)
	return o
		
def check_p(p):
	'''ensures p is an participant object and correctly loaded.
	p 	int (1-48) or participant object, if sessions are not loaded this function will do so
	'''
	if type(p) == int or not hasattr(p,'so'):
		if type(p) != int: p = p.pp_id
		p = e.Participant(p)
		p.add_all_sessions()
	return p
	
def pxml(xml):
	'''print xml to screen.'''
	print(etree.tostring(xml, encoding = 'utf8',pretty_print=True).decode())

