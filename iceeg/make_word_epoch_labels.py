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
	


def handle_block(b, force_make = False):
	if os.path.isfile(path.word_epoch_annotation + b.make_name()):
		if not force_make: 
			print('file exists, returning previously saved file.''')
			return open(path.word_epoch_annotation + b.make_name()).read().split('\n')
	b.extract_words()
	channel_corrector = get_corrector(b,'channel')
	artifact_corrector = get_corrector(b)
	block_status = b.xml.usability if hasattr(b,'xml') else 'unk'
	output = []
	for i,w in enumerate(b.words):
		word = w.word_utf8_nocode_nodia()
		word_status = 'artifact' if hasattr(w,'artifact_id') else 'clean'
		artifact_coder = w.artifact_coder  if hasattr(w,'artifact_coder') else 'unk'
		usable = str(w.usable)
		word_number = str(i)
		block_name = b.make_name()
		dirty_threshold = 'artifact' if i not in b.word_indices else 'clean'
		l = [block_name,block_status,word_number,usable,word_status,artifact_coder,word,dirty_threshold]
		output.append('\t'.join(l))
	with open(path.word_epoch_annotation + block_name,'w') as fout:
		fout.write('\n'.join(output))
	return output

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

def handle_all(force_make = False):
	output = []
	bad_blocks = []
	for i in range(1,49):
		print(i)
		p = e.Participant(i)
		p.add_all_sessions()
		for b in p.blocks:
			try: 
				output.extend(handle_block(b, force_make))
				print('-'*100,'\n'*9,b.make_name(),'\t\t\tsucces','\n'*9,'-'*100)
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

'''
block_name\tblock_quality\tword_number_in_block\tusable\tcorrected_label\tcorrector\tword\tthreshold_label\tauto_label\auto_with_threshold\tcorrected_with_threshold
'pp1_exp-o_bid-1\tgreat\t0\tTrue\tclean\tunk\thoofdstuk\tclean\tclean\tartifact\tclean'

block_name\tblock_quality\tword_number_in_block\tusable\tcorrected_label\tcorrector\tword\tthreshold_label\tauto_label
'pp10_exp-ifadv_bid-1\tgreat\t0\tFalse\tclean\tunk\tasking\tartifact\tclean'


block_name\tblock_quality\tword_number_in_block\tusable\tcorrected_label\tcorrector\tword\tthreshold_label
'pp10_exp-ifadv_bid-1\tgreat\t0\tFalse\tclean\tunk\tasking\tartifact\tclean'
'''
