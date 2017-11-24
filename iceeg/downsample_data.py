import experiment as e
import numpy as np
import os
import path

'''Downsample all data from 1000hz to 100hz sf
The code uses MNE function to downsample
Visually checked time domain and frequency domain both looked reasonable.
100hz -> nyquist at 50hz, which is ac frequency (spike in the spectrum),
however this is comfortably far from 30hz lowpass filter on the data

I do not know whether MNE uses zerophase filtering before donwsample should check.
'''


new_sf = 100
error = []


p = e.Participant(1)
fo = p.fid2ort
for i in range(1,49):
	print('loading participant:',i)
	p = e.Participant(i,fid2ort = fo)
	p.add_all_sessions()
	for s in p.sessions:
		print(s.exp_type)
		for b in s.blocks:
			print(b.bid)
			name = path.eeg100hz + b.block2name()
			print(name, name + '.npy')
			if not os.path.isfile(name + '.npy') and not b.block_missing:
				b.load_eeg_data(sf = new_sf)
				if b.raw != 0:
					print('saving data to:',name)
					np.save(name, b.raw[:][0])
				else:
					print('could not load data')
					error.append(name)
			elif b.block_missing: 
					print('block missing')
					error.append(name)
			else:
				print('file already exists, skipping:', b.block2name())
	print('-'*50)
	print('error log')
	print('\n'.join(error))
	print('-'*50)

print(error)
fout = open(path.data + 'eeg100hz_error.log','w')
fout.write('\n'.join(error))
fout.close()

