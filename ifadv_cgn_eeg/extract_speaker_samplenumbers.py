# python3
# location: /vol/tensusers/mbentum/EEG_DATA_ifadv_cgn
# use ipython to browse through objects
# use objectname? to read docstrings

import experiment



def make_output(s,pp_id,exp):
	'''Make list of sentence information.

	pp_id 				participant number
	exp 				name of the experiment
	fid 				id of audio file
	sid 				id of speaker
	sentence number 	number of sentence start at 0 speaker specfic
	st 					start time in seconds from start audio file
	et 					end time in seconds from start audio file
	st_sample 			sample number corresponding to start sentence in EEG DATA
	et_sample 			sample number corresponding to end sentence in EEG DATA
	'''
	s.set_samplenumbers()
	return [pp_id,exp,s.fid,s.sid,str(s.sentence_number),str(s.st),str(s.et),str(s.st_sample),str(s.et_sample)]

def write_output(o,filename = 'default'):
	'''Write list of lists to filename.'''
	if not filename.endswith('.txt'): filename += '.txt'
	print('Saving information to:',filename)
	fout = open(filename,'w')
	fout.write('\n'.join(['\t'.join(line) for line in o]))
	fout.close()

# load structure (fid2ort) with start and end times of word / chunks / sentences
# prints errors to the screen
e = experiment.Experiment()

# create pp 1 object with a copy of fid2ort
e.add_participant(1)

# create ifadv session with start and end sample numbers for words and sentences
e.pp1.add_session('ifadv')

# print session info
print(e.pp1.sifadv)

# set pp_id and experiment name
pp_id = str(e.pp1.sifadv.pp_id)
exp = e.pp1.sifadv.exp_type

#create a list of sentence information
output = []
for b in e.pp1.sifadv.blocks:
	print('-'*33)
	print(b)
	for s in b.sentences:
		if not s.overlap:
			output.append(make_output(s,pp_id,exp))


write_output(output,'pp1_ifadv_sentences')
			

