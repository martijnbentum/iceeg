import glob
import path


#create a dataset of all pmn_datawords for all gates


def extract_columns(line,columns,header):
	output = []
	for c in columns:
		output.append(line[header.index(c)])
	return output

def make_ds(columns = [],gates = []):
	if gates == []:
		gates = [str(g) for g in range(110,660,20)] 
	if columns == []:
		columns = 'word,exp,duration,content_word,gate,ud_entropy,entropy'
		columns += ',cross_entropy,freq_log,word_in_sentence,surprisal,ud_surprisal,word_block_index'
		columns = columns.split(',')
	header = open(path.pmn_datasets + 'header').read().strip('\n').split('\t')
	output = [columns]
	for g in gates:
		fn = glob.glob(path.pmn_datasets + str(g) + '/WORDS/*')
		print(g,len(fn))
		for f in fn:
			line = open(f).read().strip('\n').split('\t')
			output.append(extract_columns(line,columns,header))
	with open(path.pmn_datasets + 'ce_dataset','w') as fout:
		fout.write('\n'.join(['\t'.join(line) for line in output]))
	return output
	
		
	
	
