
import experiment as e
import numpy as np
import os
import path
import random
import xml_cnn
import windower


def make(fo):
	print('making event info xml files per block')
	nartifacts= 0
	nclean = 0

	fout = open('artifact_info.txt','w')
	for i in range(1,49):
		p = e.Participant(i,fid2ort = fo)
		p.add_all_sessions()
		for s in p.sessions:
			for b in s.blocks:
				if not os.path.isfile(path.eeg100hz + windower.make_name(b) +'.npy'):
					continue
				if b.start_marker_missing or b.end_marker_missing:
					d = load_100hz_numpy_block(windower.make_name(b))
					w = windower.Windower(b,nsamples= d.shape[1], sf = 100)
				w = windower.Windower(b,sf = 100)
				if not os.path.isfile(path.artifact_data_all_pp+ w.name + '_pred.npy'):
					print(path.artifact_data_all_pp +w.name + '_pred.npy', 'no prediction file present')
					continue
				if b.exp_type == 'k': nepoch =20
				if b.exp_type == 'o': nepoch =60
				if b.exp_type == 'ifadv': nepoch =80
				ii_xml = xml_cnn.xml_cnn(w,select_nartifact =nepoch, select_nclean = nepoch,cnn_model_name = 'cnn_5X5_2048_1')
				ii_xml.make_index_info()
				ii_xml.make_selection()
				ii_xml.write()
				nartifacts += ii_xml.nartifact_indices
				nclean += ii_xml.nclean_indices
				print(w.name + '\t' + ii_xml.nclean + '\t' + ii_xml.nartifact +'\n')
				fout.write(w.name + '\t' + ii_xml.nclean + '\t' + ii_xml.nartifact +'\n')

	fout.write('all_blocks\t'+str(nclean)+'\t'+str(nartifacts)+'\n')
	fout.close()


def load_100hz_numpy_block(name):
	return np.load(path.eeg100hz + name + '.npy')
