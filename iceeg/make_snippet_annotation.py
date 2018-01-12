import experiment as e
import glob
import numpy as np
import path
import windower
# import xml_snippets_artifact as xsa


def filename2block(f,fo = None):
	'''Return block object that correspond to the bad_epoch.'''
	pp_id = int(f.split('pp')[-1].split('_')[0])
	exp_type = f.split('exp-')[-1].split('_')[0]
	bid = f.split('bid-')[-1].split('.')[0]
	p = e.Participant(pp_id,fid2ort = fo)
	p.add_session(exp_type)
	s = getattr(p,'s' + exp_type)
	return getattr(s, 'b' + bid)

def load_windower(f,fo):
		b = filename2block(f,fo)
		w = windower.Windower(b,window_overlap_percentage=.99,sf=100)
		w.make_ca_info_matrix(True)
		return w

def make_indices(w):
	indices = []
	for i,line in enumerate(w.info_matrix):
		if line[4] >= .5: indices.append(1) # perc artifact overlap
		elif line[3] > 0: indices.append(0) # perc clean overlap
		elif line[5] > 0: indices.append(2) # perc other overlap
		else: raise ValueError('if not artifact should be either clean or other',i,line)	
	return indices
	

def save_info_matrix(w):
	np.save(path.snippet_annotation + w.name + '.index_info',w.info_matrix)

def save_indices(indices,w):
	indices = np.array(indices)
	np.save(path.snippet_annotation + w.name + '.gt_indices',indices)

def get_xml_files():
	fn = glob.glob(path.artifacts_clean + '*.xml')
	return fn

def make_info_matrices(fo = None,save_info = True,save_ind = True):
	fn = get_xml_files()
	for f in fn:
		w = load_windower(f,fo)
		indices = make_indices(w)
		if save_info: save_info_matrix(w)
		if save_ind: save_indices(indices,w)
