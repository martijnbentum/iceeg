import lxml
from lxml import etree
import progressbar as pb
import string
import threading


class herd():
	def __init__(self,identifier = 'all',country = None, minimum_quality = 'b'):
		self.cows = []
		self.threads = []

		for i in range(1,8):
			if i == 7: show_bar = True
			else: show_bar = False
			self.cows.append(cow(number = i, country =country, identifier = identifier, minimum_quality = minimum_quality,show_bar = show_bar))
			self.threads.append(threading.Thread(target = self.cows[-1].handle_corpus))

	def start(self):
		for t in self.threads:
			t.start()

	def show(self):
		for cow in self.cows:
			print(cow.name, cow.number)
			

class cow():
	def __init__(self,number = 1,verbose = False, country = None, minimum_quality = 'b',show_bar = True, identifier = ''):
		self.number = number
		self.verbose = verbose
		if type(country) == str: self.country = [country]
		else: self.country = country
		self.minimum_quality = minimum_quality
		self.show_bar = show_bar
		if identifier: self.identifier = identifier + '_' + minimum_quality + '_'
		else: self.identifier = minimum_quality + '_'
		self.name = 'nlcow14ax0'
		self.corpus_dir = '/vol/bigdata/corpora/COW/NLCOW/'
		self.ndict = {1:36983299,2:37167156,3:35800542,4:36737463,5:37757393,6:37174242,7:38097865}
		self.nsentences = self.ndict[number]
		self.set_conditions()

	def set_conditions(self):
		letters = [l for l in string.ascii_lowercase[:string.ascii_lowercase.index(self.minimum_quality) + 1]]
		self.ks = ['bdc','bpc']
		self.vs = [letters,letters]
		if self.country and type(self.country) == list:
			self.vs.append(self.country)
			self.ks.append('country')
		if self.verbose: print('quality_settings:',self.vs)


	def make_filename(self,identifier):
		f = self.identifier + identifier + '_'+self.name + '_' + str(self.number)
		if self.verbose: print(f)
		return f


	def open_corpus(self):
		self.f = self.corpus_dir + self.name + str(self.number) + '/' + self.name + str(self.number) + '.xml'
		if self.verbose:
			print('loading: ',self.f)
		self.corpus = open(self.f, encoding='utf-8')


	def handle_sentence(self,sentence):
		xml = self.sentence2xml(sentence)
		if self.filter_sentence(xml):
			for linetype in ['sentence','pos','lemma']:
				self.handle_line(linetype,xml)


	def handle_line(self,linetype,xml):
		line = self.make_line(xml,linetype)
		f = self.make_filename(linetype)
		self.write(f,line)
		

	def extract_sentence(self):
		sentence = []
		while 1:
			line = self.corpus.readline()
			sentence.append(line)
			if line == '</s>\n': 
				s =''.join(sentence)
				# if self.verbose: print(s)
				return s


	def sentence2xml(self,sentence):
		return etree.fromstring(sentence)


	def filter_sentence(self,xml):
		add = True
		out = ' '.join([xml.attrib[k] for k in self.ks])
		f = self.make_filename('quality')
		self.write(f,out)
		for i, _ in enumerate(self.ks):
			if xml.attrib[self.ks[i]] not in self.vs[i]: add = False
		if self.verbose:
			# print(i,ks[i],xml.attrib[self.ks[i]], self.vs[i])
			print(out,add,self.make_line(xml))
		return add


	def make_line(self,xml,line_type = 'sentence'):
		d = {'sentence':0,'pos':1,'lemma':2}
		i = d[line_type]
		return ' '.join([line.split('\t')[i] for line in xml.text.split('\n') if line])


	def write(self,name,line):
		if self.verbose: print('writing to:',name)
		if not hasattr(self,name): setattr(self,name,open(name,'w',encoding='utf-8'))
		fout = getattr(self,name)
		fout.write(line + '\n')
				

	def handle_corpus(self):
		self.open_corpus()
		self.index = 0
		if self.show_bar:
			bar = pb.ProgressBar()
			bar(range(self.nsentences))
		while 1:
			sentence = self.extract_sentence()
			# print([sentence])
			self.handle_sentence(sentence)
			self.index += 1
			if self.show_bar:
				bar.update(self.index)
			if self.verbose: print(self.index)
			if self.index > self.nsentences: break
			if sentence == None: break
		
		
def handle_cow(c):
	c.open_corpus()
	c.index = 0
	if c.show_bar:
		bar = pb.ProgressBar()
		bar(range(c.nsentences))
	while 1:
		sentence = c.extract_sentence()
		# print([sentence])
		c.handle_sentence(sentence)
		c.index += 1
		if c.show_bar:
			bar.update(c.index)
		if c.verbose: print(c.index)
		if c.index > c.nsentences: break
		if sentence == None: break

		
		


	



