
import experiment as e
import numpy as np
import os
import path
import random
import xml_cnn
import windower


def make(fo):
	print('making artifacts, cnn bad epoch, xml files per block')
	cnn_model_name = 'cnn_5X5_2048_1'

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
				a = xml_cnn.xml_cnn(w)
				a.make_bad_epoch()
				a.bad_epochs2xml()
				a.write_bad_epoch_xml()
				print(w.name) 



def load_100hz_numpy_block(name):
	return np.load(path.eeg100hz + name + '.npy')
