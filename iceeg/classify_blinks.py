import blinks
import experiment
import glob
import load_all_ort
import os
import path
import pickle
import random


def make_blinks():
	'''Make all blinks objects and store them as pickled blinks objects.'''
	for i in range(1,49):
		print('-'*50)
		print('-'*50)
		print('make blinks participant:',i)
		print('-'*50)
		print('-'*50)
		if not os.path.isfile(path.blinks + 'pp'+str(i)+'.done'):
			p = experiment.Participant(i)
			p.add_all_sessions()
			for s in p.sessions:
				print(s)
				for b in s.blocks:
					b.make_blinks(False)
			fout = open(path.blinks + 'pp'+str(i)+'.done','w')
			fout.close()
		else:
			print('skipping participent:',i,'already done')

def manually_classify_blinks():
	'''Load blink file and show all possible blinks detected with peakutils (see blink.py)
	and save as text file with participant and experiment information
	1 blink
	2 no blink
	(will be recoded to 1 blink 0 no blink, this was easier during classification)
	classification files have the same filename, however they have marker code twice due
	to coding error.
	'''

	# name = input('name: ')
	name = 'martijn'
	end = '_' + name + '.classification'

	fn = glob.glob(path.blinks + '*.blinks')
	done = glob.glob(path.blinks + '*'+ end)
	f_done = [f for f in fn if f.strip('.blinks')+ '_' +f.strip('.blinks').split('_')[-1] +end in done]


	print('-'*50)
	print(len(fn),'blink files ')
	print(len(done),'files classified by ',name)
	print(len(fn)-len(done), 'files remaining')
	print(len(f_done),len(done))

	random.shuffle(fn)
	i =1
	for f in fn:
		if f not in f_done:
			print('classifying:',f)
			fin = open(f,'rb')
			b = pickle.load(fin)
			b.classify_blinks()
		else: print('skipping file:',f,'already done')
		print('this was file:',i,'during this session')
		i+=1


def model_classify_blinks(exp = None,model_name = '', fo = None,start_index = 0):
	if model_name == '': print('using default blink model.')
	if exp == None:
		exp = experiment.Experiment()
		if fo == None: fo = load_all_ort.load_fid2ort()
		exp.add_all_participants(fid2ort = fo)
		exp.add_all_sessions()

	skipped_blocks = []
	for b in exp.blocks[start_index:]:
		blink = blinks.Blinks(b)
		succes = blink.model_classification(save = True)
		if succes == 0: skipped_blocks.append(b)
		b.unload_eeg_data()


	
	
