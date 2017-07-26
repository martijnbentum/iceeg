import blinks
import glob
import path
import pickle
import random

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


