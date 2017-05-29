import glob



'''
python3
retrieves word onsets from filename in log file
could be extended to retrieve phoneme onsets
'''


def read_log(f):
	# reads in the log file
	return [line.split(' ') for line in open(f).read().split('\n')]

def get_wav_fn(log):
	# get all wav names from a log file
	return [line[6] for line in log if line != ['']]

def wav_f2table_f(wav_f,table_fn):
	# finds the corresponding filename based on wav filename and list of filenames
	# id part of wav file name must uniquely correspond with one of the list of filenames
	wav_id = wav_f.split('_')[0]
	for line in table_fn:
		if wav_id in line:
			return line
	print('table filename not found for: ',wav_f)
	return None

def read_table(f):
	# reads a praat table (export from textgrid)
	print(f)
	return [line.split('\t') for line in open(f).read().split('\n') if line]


def extract_word(table):
	# extracts the word from the awd table and return start and end time and tier name
	output = []
	exclusion_list = ['sp','sil']
	for line in table:
		tier_info = line[1].split('-')
		if len(tier_info) > 2 and tier_info[1] == 'ort' and tier_info[2] == 'word':
			word = line[2]
			if word not in exclusion_list:
				start_time = str(int(float(line[0]) * 1000))
				end_time = str(int(float(line[-1]) * 1000))
				output.append([tier_info[0],line[2],start_time,end_time])	
	return output


def get_speaker_id(wav_f):
	# find the speaker ids based on the wav file name
	recording_id = wav_f.split('_')[0]
	recordings = read_table('../IFADV_ANNOTATION/recordings_table.txt')
	header = recordings[0]
	for line in recordings[1:]:
		if line[header.index('idA')] == recording_id:
			return line[header.index('subA')], line[header.index('subB')]
	print('Did not find recording id',recording_id)
	return None

def get_speaker_info(speaker_id):
	# return gender and age of the speaker based on the idea letter of speaker
	speakers = read_table('../IFADV_ANNOTATION/subjects_table.txt')
	header = speakers[0]
	for line in speakers[1:]:
		if line[header.index('IDcode')] == speaker_id:
			return line[header.index('Sex')], line[header.index('Age')]
	print('Did not find speaker info',speaker_id)
	return None

def get_words(wav_f):
	# extract word start end time and tier (speaker) info
	table_fn = glob.glob('../IFADV_ANNOTATION/AWD/WORD_TABLES/*.Table')
	table_f = wav_f2table_f(wav_f,table_fn)
	table = read_table(table_f)
	words = extract_word(table) # excludes certain words: sp sil
	return words

def get_pos(wav_f):
	# get part of speech tags annotation
	pos_fn = glob.glob('../IFADV_ANNOTATION/POS/*.pos')
	pos_f = wav_f2table_f(wav_f,pos_fn)
	pos = open(pos_f).read().split('\n')
	return pos

def get_word_onsets(wav_f):
	# retrieve word and pos annotation, retrieve speaker info, combine in onset table
	# channel speakerid gender age word_number word start_ms end_ms
	words = get_words(wav_f)
	pos = get_pos(wav_f)

	left_id,right_id = get_speaker_id(wav_f)
	left_gender,left_age = get_speaker_info(left_id)
	right_gender,right_age = get_speaker_info(right_id)
	output = []
	for i,line in enumerate(words):
		word, st, et = line[1:]
		if line[0] == 'spreker1':
			output.append(['left',left_id,left_gender,left_age, str(i+1),word, st, et])
		if line[0] == 'spreker2':
			output.append(['right',right_id,right_gender,right_age, str(i+1),word, st, et])
	return output

def get_pos_tag(p,word_info):
	speaker_dict = {'left':'spreker1','right':'spreker2'}
	start_time = float(word_info[-2]) / 1000.0
	word = word_info[-3]
	speaker = speaker_dict[word_info[0]]
	s = p.find_sentence(start_time,speaker)
	if s:
		pos_tag = s.find_word(word)
		print(s)
		print(word_info)
		print(word)
		print(pos_tag)
		if pos_tag:
			return pos_tag
		else:
			return 'NA'
	else:
		print('sentence not found',word,word_info)
		return 'NA'

	

def ptable(table):
	# easily print tables with nice layout
	print('\n'.join(['\t'.join(line) for line in table]))
	print()

import pos
# get some log filenames for testing
log_dir = '../log/'
exp_type = 'ifadv'
lfn = glob.glob(log_dir+'*'+exp_type+'*log*.txt')
# load a log file for testing
logf = lfn[0]
log = read_log(logf)
# get a wav file name for testing
wav_fn = get_wav_fn(log)
wav_f = wav_fn[1]

# getting word and pos info from ifadv annotation
words = get_words(wav_f)
p = pos.Pos(wav_f)

# final functionality get onsets based on wav filename in log file
onsets = get_word_onsets(wav_f)
# ptable(onsets)
output = []
for line in onsets:
	pos_tag = get_pos_tag(p,line)
	output.append(line + [pos_tag])

ptable(output)
# print(output)
print(wav_f)
print(p.sentences[-1])
