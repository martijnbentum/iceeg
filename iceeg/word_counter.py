import experiment as e
import path
import utils

def count_words(save = False):
	'''count words in the EEG experiment per block and exp type.
	count all presented words (all), count all presented content words (cw-all)
	count the subset of words that is useable (ok), count the cw subset (cw-ok)
	save 		whether to save the dict (default false because it exists)
	'''
	word_counter = {}
	bad_blocks = []
	output = []

	for i in range(1,49):
		p = e.Participant(i)
		pp_id = p.pp_id
		p.add_all_sessions()
		print(p)
		for s in p.sessions:
			exp_type = s.exp_type
			if exp_type not in word_counter.keys(): word_counter[exp_type] = 0
			for b in s.blocks:
				try: b.extract_words(content_word = False)
				except:
					bad_blocks.append(b)
					continue
				if not hasattr(b,'extracted_words'):
					bad_blocks.append(b)
					continue
				add_wc(b,word_counter,exp_type)
				add_wc(b,word_counter,b.name)
	if save: save_count(word_counter)
	return word_counter, bad_blocks


def add_wc(b, word_counter, identifier):
	'''Add the word count for each condition: all cw-all ok cw-ok.
	b 		block object
	wor... 	dictionary to add the counts to
	id... 	whether it is an exp_type or block to add the counts to
	'''
	if identifier +'_all' not in word_counter.keys():
		word_counter[identifier+'_all'] = len(b.words)
		word_counter[identifier+'_cw-all'] = len([w for w in b.words if hasattr(w,'pos') and w.pos.content_word])
		word_counter[identifier+'_ok'] = len(b.extracted_words)
		word_counter[identifier+'_cw-ok'] = len([w for w in b.extracted_words if hasattr(w,'pos') and w.pos.content_word])
	else:
		word_counter[identifier+'_all'] += len(b.words)
		word_counter[identifier+'_cw-all'] += len([w for w in b.words if hasattr(w,'pos') and w.pos.content_word])
		word_counter[identifier+'_ok'] += len(b.extracted_words)
		word_counter[identifier+'_cw-ok'] += len([w for w in b.extracted_words if hasattr(w,'pos') and w.pos.content_word])
	

def save_count_dict(wc):
	'''Save the word count dict to file.
	wc 		dictionary containing the word count
	'''
	output =[]
	for k in wc.keys():
		output.append(k+'\t'+str(wc[k]))
	with open(path.data + 'word_count_experiment.dict','w') as fout:
		fout.write('\n'.join(output))

def save_count_ds(wc):
	'''Save the word count dict as dataset to extract some info about block/participant based wc statistics.
	wc 		dictionary containing the word count
	'''
	output =[]
	for k in wc.keys():
		if not k.startswith('pp'): continue
		temp = k.split('_')
		bname = '_'.join(temp[:3])
		pp_id,exp_type,bid = utils.name2pp_id(bname),utils.name2exp_type(bname),utils.name2bid(bname)
		cw = 'cw' if 'cw' in temp[-1] else 'all'
		ok = 'ok' if 'ok' in temp[-1] else 'all'
		l = '\t'.join([str(pp_id),exp_type,str(bid),cw,ok,str(wc[k]),k])
		output.append(l)
	with open(path.data + 'word_count_experiment.ds','w') as fout:
		fout.write('\n'.join(output))

		
		
			
