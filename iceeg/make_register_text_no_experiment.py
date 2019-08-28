import glob
import path

def load_fid_info():
	t = open('/Volumes/storage/awd_utf8/wav-dvd-files.txt').read().split('\n')[1:-1]
	t = [line.split('\t') for line in t]
	t = [line for line in t if line if 'nl' in line[1]]
	for line in t:
		line[1] = line[1].split('-')[-1].split('/')[0]
		line[2] = line[2].split('.')[0]
	d = dict([[line[2],line[1]] for line in t])
	rd = {}
	for k in d.keys():
		comp = d[k]
		if comp not in rd.keys(): rd[comp] = []
		rd[comp].append(k)
	return t,d,rd


def open_experiment_fid():
	ifadv_fid = open(path.data +'ifadv_fids').read().split('\n')
	k_fid = open(path.data +'k_fids').read().split('\n')
	o_fid = open(path.data +'o_fids').read().split('\n')
	return ifadv_fid,k_fid,o_fid

def _clean_text(t):
	t = ' '.join([word.split('*')[0] for word in t.split(' ')])
	t = t.lower()
	t = t.replace('???','.')
	t = t.replace('??','.')
	t = t.replace('?','.')
	t = t.replace('!!!','.')
	t = t.replace('!!','.')
	t = t.replace('!','.')
	t = t.replace('...','.')
	t = t.replace('. .','.')
	t = t.replace('..','.')
	t = t.replace('.',' .\n')
	t = t.replace('  ',' ')
	t = [line.strip() for line in t.split('\n') if line != '']
	return t

def read_table(f):
	t = ' '.join([line.split('\t')[-2] for line in open(f).read().split('\n') if len(line.split('\t')) ==4][1:])
	return _clean_text(t)

def make_no_experiment_fid():
	t,d,rd = load_fid_info()
	ifadv_fid,k_fid,o_fid = open_experiment_fid()
	k_noexp_fids = list(set(rd['k']) - set(k_fid))
	o_noexp_fids = list(set(rd['o']) - set(o_fid))
	print('org k:',len(rd['k']),'exp k:',len(k_fid),'no-exp k:',len(k_noexp_fids))
	print('org o:',len(rd['o']),'exp o:',len(o_fid),'no-exp o:',len(o_noexp_fids))
	return k_noexp_fids, o_noexp_fids 

def read_cgn_fids(fids):
	fn = glob.glob(path.cgn_annot+'TABLE_CGN2_ORT/*.Table')
	text_fids= []
	not_found = []
	for fid in fids:
		found = False
		for f in fn:
			if fid in f: 
				text_fids.extend(read_table(f))
				found = True
				break
		if not found: 
			not_found.append(fid)
			print('did not find',fid)
	return text_fids, not_found

def _clean_ifadv(filename):
	t = open(filename).read().replace('\n',' ')
	return _clean_text(t)

def read_ifadv(experimental_fids):
	fn = glob.glob(path.data + 'PREPROC_LM_TRAIN_TEST/IFADV_PREPROC/*.preproc')
	text_fids,text_exp = [],[]
	excluded = []
	for f in fn:
		found = False
		for fid in experimental_fids:
			if fid in f: found = True 
		if not found: text_fids.extend(_clean_ifadv(f))
		else: 
			text_exp.extend(_clean_ifadv(f))
			excluded.append(f)
	print('excluded these files:',excluded)
	print('experimental_fids:',experimental_fids)
	print('n excluded:',len(excluded),'n experimental:',len(experimental_fids))
	return text_fids, text_exp


def write_no_exp_text():
	ifadv_fid,k_fid,o_fid = open_experiment_fid()
	knexp, onexp = make_no_experiment_fid()
	tknexp, nfk = read_cgn_fids(knexp)
	tkexp, _ = read_cgn_fids(k_fid)
	tonexp , nfo = read_cgn_fids(onexp)
	toexp, _ = read_cgn_fids(o_fid)
	tifadvnexp, tifadvexp = read_ifadv(ifadv_fid)
	if len(nfk) == len(nfo) == 0: pass
	else: print('WARNING, for k, o, ifadv files are missing',nfk,nfo,nfi)
	with open(path.data + 'no_experiment_text_comp-k.preproc','w') as fout:
		fout.write('\n'.join(tknexp))
	with open(path.data + 'no_experiment_text_comp-o.preproc','w') as fout:
		fout.write('\n'.join(tonexp))
	with open(path.data + 'no_experiment_text_ifadv.preproc','w') as fout:
		fout.write('\n'.join(tifadvnexp))
	with open(path.data + 'experiment_text_comp-k.preproc','w') as fout:
		fout.write('\n'.join(tkexp))
	with open(path.data + 'experiment_text_comp-o.preproc','w') as fout:
		fout.write('\n'.join(toexp))
	with open(path.data + 'experiment_text_ifadv.preproc','w') as fout:
		fout.write('\n'.join(tifadvexp))
	print('saved text files to: no_experiment_text_comp-o.preproc no_experiment_text_comp-k.preprocp no_experiment_text_ifadv.preproc')
	print('and the experiment_text version without no_ prepended')

			
	

	
