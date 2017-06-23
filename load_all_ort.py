ifadv = [line.split('\t') for line in open('../fnlist_ifadv.txt').read().split('\n')]
fno = [line.split('\t') for line in open('../fnlist_o.txt').read().split('\n')]
fnk = [line.split('\t') for line in open('../fnlist_k.txt').read().split('\n')]

o_content_words = 0
k_ncontent_words = 0
ifadv_ncontent_words = 0

k = []
o =[]
ifadv = []

for line in fnk:
    k.append(ort.Ort(fid = line[1],sid_name=line[0],path ='../TABLE_CGN2_ORT/',awd_path = '../../CGN/TABLE_CGN2_AWD/',corpus='CGN',pos_path = 'POS_K/FROG_OUTPUT/',register = 'news_broadcast'))
    k_ncontent_words += k[-1].speakers[0].ncontent_words
    print(k[-1])

for line in fno:
    o.append(ort.Ort(fid = line[1],sid_name=line[0],path ='..//TABLE_CGN2_ORT/',awd_path = '../../CGN/TABLE_CGN2_AWD/',corpus='CGN',pos_path = 'POS_O/FROG_OUTPUT/',register = 'read_aloud_stories'))
    o_ncontent_words += o[-1].speakers[0].ncontent_words
    print(o[-1])

for line in fnifadv:
    ifadv.append(ort.Ort(fid = line[2],sid_name=line[0],path ='../IFADV_ANNOTATION/ORT/',awd_path = '../IFADV_ANNOTATION/AWD/WORD_TABLES/',corpus='IFADV',pos_path = 'POS_IFADV/FROG_OUTPUT/',register = 'spontaneous_dialogue'))
    ifadv[-1].add_speaker(line[1])
	ifadv[-1].check_overlap()
    print(ifadv[-1])
    ifadv_ncontent_words += ifadv[-1].speakers[0].ncontent_words
    ifadv_ncontent_words += ifadv[-1].speakers[1].ncontent_words

print('ifadv ncontent',ifadv_ncontent_words)
print('k ncontent',k_ncontent_words)
print('o ncontent',o_ncontent_words)
print('all ncontent',ifadv_ncontent_words+k_ncontent_words+o_content_words)
print('all times 48 pp',(ifadv_ncontent_words+k_ncontent_words+o_content_words)*48)


