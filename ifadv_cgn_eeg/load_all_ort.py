def make_fid2ort(verbose = False):
	import ort
	'''Make fid2ort dictionary that maps file id to ort objects.

	deepcopying this object takes approx 18 seconds while making it takes
	90 seconds. Options is to make it once and copy it for each participant
	'''

	fnifadv = [line.split('\t') for line in open('../fnlist_ifadv.txt').read().split('\n')]
	fno = [line.split('\t') for line in open('../fnlist_o.txt').read().split('\n')]
	fnk = [line.split('\t') for line in open('../fnlist_k.txt').read().split('\n')]

	o_ncontent_words = 0
	k_ncontent_words = 0
	ifadv_ncontent_words = 0

	k = {}
	o = {}
	ifadv = {}
	fid2ort = {}

	for line in fnk:
		fid2ort[line[1]] = ort.Ort(fid = line[1],sid_name=line[0],path ='../TABLE_CGN2_ORT/',awd_path = '../../CGN/TABLE_CGN2_AWD/',corpus='CGN',pos_path = 'POS_K/FROG_OUTPUT/',register = 'news_broadcast')
		k_ncontent_words += fid2ort[line[1]].speakers[0].ncontent_words
		if verbose:
			print(fid2ort[line[1]])

	for line in fno:
		fid2ort[line[1]] = ort.Ort(fid = line[1],sid_name=line[0],path ='..//TABLE_CGN2_ORT/',awd_path = '../../CGN/TABLE_CGN2_AWD/',corpus='CGN',pos_path = 'POS_O/FROG_OUTPUT/',register = 'read_aloud_stories')
		o_ncontent_words += fid2ort[line[1]].speakers[0].ncontent_words
		if verbose:
			print(fid2ort[line[1]])

	for line in fnifadv:
		fid2ort[line[2]] = ort.Ort(fid = line[2],sid_name=line[0],path ='../IFADV_ANNOTATION/ORT/',awd_path = '../IFADV_ANNOTATION/AWD/WORD_TABLES/',corpus='IFADV',pos_path = 'POS_IFADV/FROG_OUTPUT/',register = 'spontaneous_dialogue')
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

