import experiment as e
import glob
import os
import ppl
import path
import utils

def get_corrector(b, correction_type = 'artifact'):
	if correction_type == 'artifact':
		d = path.corrected_artifact_cnn_xml 
	elif correction_type == 'channel':
		d = path.corrected_ch_cnn_xml
	else: raise ValueError(correction_type,'unknown, should be artifact or channel')
	fn = glob.glob(d + '*' + b.make_name() + '.xml')
	if fn == []: return 'unk'
	if len(fn) > 1: raise ValueError(fn,'returned more filenames than expected')
	corrector = fn[0].split('/')[-1].split('-')[0]
	if corrector == 'rep': return 'model-' + correction_type
	if corrector not in 'martijn,tim'.split(','): 
		raise ValueError(corrector,fn,'unexpected value (expected tim or martijn)')
	return corrector + '-' + correction_type
	


def handle_block(b, force_make = False, ppls = None):
	if os.path.isfile(path.word_epoch_annotation + b.make_name()):
		if not force_make: 
			print('file exists, returning previously saved file.''')
			return open(path.word_epoch_annotation + b.make_name()).read().split('\n')
	if ppls == None: ppls = ppl.word2ppl(exp = b.exp_type)
	block_name = b.make_name()
	bua = utils.load_block_with_uncorrected_artifacts(block_name)
	b_dirty = utils.name2block(block_name)
	ppls.add_ppl_to_words(bua)
	bua.extract_words(content_word = False)
	b.extract_words(content_word = False)
	b_dirty.extract_words(content_word = False,dirty = True)
	channel_corrector = get_corrector(b,'channel')
	artifact_corrector = get_corrector(b)
	block_status = b.xml.usability if hasattr(b,'xml') else 'unk'
	output = []
	for i,w in enumerate(b.words):
		if hasattr(w,'pos') and w.pos.content_word: content_word = 'cw'
		else: content_word = 'ncw'
		word = w.word_utf8_nocode_nodia()
		word_status = 'artifact' if hasattr(w,'artifact_id') else 'clean'
		corrected_threshold = 'artifact' if i not in b.word_indices else 'clean'
		dirty_threshold = 'artifact' if i not in b_dirty.word_indices else 'clean'
		auto_threshold = 'artifact' if i not in bua.word_indices else 'clean'
		usable = str(w.usable)
		word_number = str(i)
		l = [block_name,block_status,word_number,word,usable,content_word]
		l += [word_status,corrected_threshold,dirty_threshold,auto_threshold]
		output.append('\t'.join(l))
	with open(path.word_epoch_annotation + block_name,'w') as fout:
		fout.write('\n'.join(output))
	return output

def make_header():
	'''make header for the word epoch label files (header is not part of the files.
	block name 		name of experiment block
	block status 	usability of a block (bad and doubtfull were considered unusable)
					mediocre, ok and great were considered usable
	word number 	index of the word in the block (count of words in block starting at zero)
	word 			the ortographic form of the word without diacritics or capitals
	usable 			whether the word is usable (no overlap with other words no artifacts
					phonemic transcription present (needed for alignment with eeg data)
	content word 	whether the word is a content word
	word status 	artifact or clean (whether it overlaps with an artefact) based on corrected
					automatic annotations (data ica cleaned)
	corrected_... 	same now the eeg data in the word epoch is also checked, if this exceeds
					75 mu volts the word epoch is excluded (data ica cleaned)
	dirty_thres... 	same however no annotations or ICA cleaning was used on thresholding on raw
					data (baseline procedure for classifier performance)
	auto_thres... 	same as corrected_threshold however using the uncorrected annotation to test
					the performance of classifier only		
	'''
	l = 'block_name,block_status,word_number,word,usable,content_word'
	l += ',word_status,corrected_threshold,dirty_threshold,auto_threshold'
	output = l.split(',')
	output = '\t'.join(output)
	with open(path.word_epoch_annotation + 'header','w') as fout:
		fout.write(output)
	return output
	
	


def handle_all(force_make = False):
	output = []
	bad_blocks = []
	for i in range(1,49):
		print(i)
		p = e.Participant(i)
		p.add_all_sessions()
		for s in p.sessions:
			for b in s.blocks:
				try: 
					output.extend(handle_block(b, force_make,ppls = s.ppl ))
					print('-'*100,'\n'*9,b.make_name(),'\t\t\tsucces','\n'*9,'-'*100)
				except: 
					bad_blocks.append(b)
					print('!'*100,'\n'*9,b.make_name(),'\t\t\tfailed','\n'*9,'!'*100)
	return output,bad_blocks
			
	


'''
block_name\tblock_quality\tword_number_in_block\tusable\tcorrected_label\tcorrector\tword\tthreshold_label\tauto_label\auto_with_threshold\tcorrected_with_threshold
'pp1_exp-o_bid-1\tgreat\t0\tTrue\tclean\tunk\thoofdstuk\tclean\tclean\tartifact\tclean'

block_name\tblock_quality\tword_number_in_block\tusable\tcorrected_label\tcorrector\tword\tthreshold_label\tauto_label
'pp10_exp-ifadv_bid-1\tgreat\t0\tFalse\tclean\tunk\tasking\tartifact\tclean'


block_name\tblock_quality\tword_number_in_block\tusable\tcorrected_label\tcorrector\tword\tthreshold_label
'pp10_exp-ifadv_bid-1\tgreat\t0\tFalse\tclean\tunk\tasking\tartifact\tclean'

def add_labelling_based_on_automatic_annotations(b,save = False):
	block_name = b.make_name()
	o = handle_block(b)
	if len(o[0].split('\t')) == 9:
		print('labelling already added?, expected 8 columns, got 9\n\n',o[0],'\n\n')
		return 0
	b = utils.load_block_with_uncorrected_artifacts(block_name)
	for i,w in enumerate(b.words):
		word_status = 'artifact' if hasattr(w,'artifact_id') else 'clean'
		o[i] += '\t'+word_status
	if save:
		with open(path.word_epoch_annotation + block_name,'w') as fout:
			fout.write('\n'.join(o))
	return o


def add_labelling_with_threshold(b,save = False, ppls = None):
	if ppls == None: ppls = ppl.word2ppl(exp = b.exp_type)
	block_name = b.make_name()
	o = handle_block(b)
	if len(o[0].split('\t')) == 11:
		print('labelling already added?, expected 9 columns, got 11\n\n',o[0],'\n\n')
		return 0
	bua = utils.load_block_with_uncorrected_artifacts(block_name)
	ppls.add_ppl_to_words(bua)
	bua.extract_words(content_word = False)
	b.extract_words(content_word = False)
	for i,w in enumerate(bua.words):
		if hasattr(w,'pos') and w.pos.content_word: content_word = 'cw'
		else: content_word = 'ncw'
		auto_threshold = 'artifact' if i not in bua.word_indices else 'clean'
		corrected_threshold = 'artifact' if i not in b.word_indices else 'clean'
		o[i] += '\t'+auto_threshold + '\t' + corrected_threshold + '\t' + content_word
	if save:
		with open(path.word_epoch_annotation + block_name,'w') as fout:
			fout.write('\n'.join(o))
	return o


def add_all_labelling_with_threshold():
	output = []
	bad_blocks = []
	for i in range(1,49):
		print(i)
		p = e.Participant(i)
		p.add_all_sessions()
		for s in p.sessions:
			for b in s.blocks:
				try: 
					o = add_labelling_with_threshold(b, True, ppls = s.ppl)
					if o == 0: continue
					output.extend(o)
					print('-'*100,'\n'*9,b.make_name(),'\t\t\t---succes---','\n'*9,'-'*100)
				except: 
					bad_blocks.append(b)
					print('!'*100,'\n'*9,b.make_name(),'\t\t\tfailed','\n'*9,'!'*100)
	return output,bad_blocks


def add_all_labelling():
	output = []
	bad_blocks = []
	for i in range(1,49):
		print(i)
		p = e.Participant(i)
		p.add_all_sessions()
		for b in p.blocks:
			try: 
				o = add_labelling_based_on_automatic_annotations(b, True)
				if o == 0: continue
				output.extend(o)
				print('-'*100,'\n'*9,b.make_name(),'\t\t\tsucces','\n'*9,'-'*100)
			except: 
				bad_blocks.append(b)
				print('!'*100,'\n'*9,b.make_name(),'\t\t\tfailed','\n'*9,'!'*100)
	return output,bad_blocks
'''

