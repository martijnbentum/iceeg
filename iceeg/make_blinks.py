import experiment
import os
import path

for i in range(1,49):
	print('-'*50)
	print('-'*50)
	print('Detecting blinks participant:',i)
	print('-'*50)
	print('-'*50)
	if not os.path.isfile(path.blinks + 'pp'+str(i)+'.done'):
		p = experiment.Participant(i)
		p.add_all_sessions()
		for s in p.sessions:
			print(s)
			for b in s.blocks:
				b.detect_blinks(False)
		fout = open(path.blinks + 'pp'+str(i)+'.done','w')
		fout.close()
	else:
		print('skipping participent:',i,'already done')
