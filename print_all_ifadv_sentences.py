import glob
import ort

def get_all_fid_ifadv():
	directory = '/Users/Administrator/storage/EEG_DATA_ifadv_cgn/IFADV_ANNOTATION/ORT/TABLE/'
	fn = glob.glob(directory + 'DVA*')
	return [f.split('/')[-1].split('_')[0] for f in fn]

for fid in get_all_fid_ifadv():
	print('loading:',fid)
	o = ort.Ort(fid = fid)
	o.add_speaker('spreker2')
	o.make_sentences()
	o.print_sentences()
	

