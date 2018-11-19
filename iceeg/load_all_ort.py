import path
import pickle

def make_fid2ort(verbose = False):
	import ort
	'''Make fid2ort dictionary that maps file id to ort objects.

	deepcopying this object takes approx 18 seconds while making it takes
	90 seconds. Options is to make it once and copy it for each participant
	'''

	print(path.data)

	fnifadv = [line.split('\t') for line in open(path.data + 'fnlist_ifadv.txt').read().split('\n')]
	fno = [line.split('\t') for line in open(path.data + 'fnlist_o.txt').read().split('\n')]
	fnk = [line.split('\t') for line in open(path.data + 'fnlist_k.txt').read().split('\n')]

	o_ncontent_words = 0
	k_ncontent_words = 0
	ifadv_ncontent_words = 0

	k = {}
	o = {}
	ifadv = {}
	fid2ort = {}

	for line in fnk:
		fid2ort[line[1]] = ort.Ort(fid = line[1],sid_name=line[0],path = path.cgn_ort,awd_path = path.cgn_awd,corpus='CGN',pos_path = path.compk_pos,register = 'news_broadcast',set_verbose = verbose)
		k_ncontent_words += fid2ort[line[1]].speakers[0].ncontent_words
		if verbose:
			print(fid2ort[line[1]])

	for line in fno:
		fid2ort[line[1]] = ort.Ort(fid = line[1],sid_name=line[0],path=path.cgn_ort,awd_path=path.cgn_awd,corpus='CGN',pos_path = path.compo_pos,register = 'read_aloud_stories',set_verbose = verbose)
		o_ncontent_words += fid2ort[line[1]].speakers[0].ncontent_words
		if verbose:
			print(fid2ort[line[1]])

	for line in fnifadv:
		fid2ort[line[2]] = ort.Ort(fid = line[2],sid_name=line[0],path =path.ifadv_ort,awd_path = path.ifadv_awd,corpus='IFADV',pos_path = path.ifadv_pos,register = 'spontaneous_dialogue',set_verbose = verbose)
		fid2ort[line[2]].add_speaker(line[1])
		fid2ort[line[2]].check_overlap()
		if verbose:
			print(fid2ort[line[2]])
		ifadv_ncontent_words += fid2ort[line[2]].speakers[0].ncontent_words
		ifadv_ncontent_words += fid2ort[line[2]].speakers[1].ncontent_words

	if verbose:
		print('ifadv ncontent',ifadv_ncontent_words)
		print('k ncontent',k_ncontent_words)
		print('o ncontent',o_ncontent_words)
		print('all ncontent',ifadv_ncontent_words+k_ncontent_words+o_ncontent_words)
		print('all times 48 pp',(ifadv_ncontent_words+k_ncontent_words+o_ncontent_words)*48)
	return fid2ort

def save_fid2ort(fid2ort):
	'''Save fid2ort in a pickle to the data directory.'''
	fout = open(path.data + 'fid2ort.dict','wb')
	pickle.dump(fid2ort,fout,-1)
	fout.close()

def load_fid2ort():
	'''Load the fid2ort dictionary.'''
	fin = open(path.data + 'fid2ort.dict','rb')
	fid2ort = pickle.load(fin)
	fin.close()
	return fid2ort
