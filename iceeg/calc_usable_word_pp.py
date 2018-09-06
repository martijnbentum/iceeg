import combine_artifacts as ca
import experiment as e
import os
import path
import utils


def write(line, exp, purge = False):
	f = 'usable_words_per_participant_' + exp + '.txt'
	if purge: open(path.data + f,'w').close()
	with open(path.data + f,'a') as fout:
		fout.write(line)


def read(exp):
	f = 'usable_words_per_participant_' + exp + '.txt'
	return [line.split('\t')[0] for line in open(path.data + f).read().split('\n')[:-1]]

	
def run(exp = 'o', force_new = False):
	if force_new:
		write('',exp,purge = True)
	present_names = read(exp)
	names =  ca.read_names(exp)
	for name in names:
		print(name)
		if name in present_names and not force_new:
			print('name already present, skipping to following.')
			continue
		b = utils.name2block(name)
		# if os.path.isfile(path.bads_annotations + 'bads_' + b.name + '.xml'):
		u = b.xml.usability 
		p = round(b.xml.clean_perc,2)
		if hasattr(b,'xml') and u not in ['bad','doubtfull']:
			line = [b.name,b.nallwords,b.nwords,b.ncontent_words,b.ncwords,u,p]
			line = map(str,line)
			line = '\t'.join(line) + '\n'
			print(line)
			write(line,exp)
		del b
	return missing
				
			
				
		
	
	
	
