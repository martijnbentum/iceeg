import glob
import os
import socket
import time
import make_lexicon_word_prob as mlwp


def reset_files(file_infos):
	for fi in file_infos:
		for d in ['DONE_LT/','LEXICON_TXT/','DONE_PPL/','FINISHED_PPL/','PPL/','DONE_PDF/']:
			command = 'rm '+d +fi.name + '*'
			print(command)
			os.system(command)
	for fi in file_infos:
		os.system('touch PROBLEMS/' + fi.name)

def handle_lt(text,max_nthreads=12, lt_dir = 'LEXICON_TXT/',lt_done ='DONE_LT/',lexicon = None, max_i = None):
	with open('working_ponies','a') as fout: fout.write(socket.gethostname() + '\tlt\t' +  time.strftime('%Y-%m-%d %H:%M:%S'))
	if lexicon == None: lexicon = mlwp.load_lexicon()
	tm = mlwp.thread_master(max_nthreads)
	i = 0

	while 1:
		# check whether to stop processing
		if os.path.isfile('stop_lt') or i == max_i: break

		# check whether there is thread available and no pause is set
		if tm.thread_available('pdf') and not os.path.isfile('pause_lt'):
			sentence = text[i].replace("'",'').replace('"','')
			job_accepted = tm.add_thread('lt',mlwp.make_lexicon_text,(sentence,lexicon,lt_dir))
			if job_accepted: print('processing sentence:',sentence)
			if job_accepted and i < len(text): i += 1

		elif os.path.isfile('paused_lt'): print('paused')
		else: print('waiting... for available threads')
		time.sleep(1)
		print('currently at index:',i,'of:',len(text),'\n',tm)

	with open('working_ponies','a') as fout: fout.write(socket.gethostname()+' done lt\t'+time.strftime('%Y-%m-%d %H:%M:%S'))
	print(socket.gethostname() + 'done lt', time.strftime('%Y-%m-%d %H:%M:%S'))



def handle_pdf(offset = 0, max_nthreads=20,lexicon = '',ppl_dir = 'PPL/',pdf_dir = 'PDF/',pdf_done = 'DONE_PDF/', max_i =None):
	with open('working_ponies','a') as fout: fout.write(socket.gethostname() + '\tpdf\t' + str(offset) +'\n')
	if lexicon == '': lexicon = mlwp.load_lexicon()
	tm = mlwp.thread_master(max_nthreads)
	print(tm)
	finished = False
	i = 0

	while 1:
		# check whether to stop processing
		if os.path.isfile('stop_pdf') or max_i == i: break
		i += 1 

		print('creating file matcher to see what needs to be done. Index:',i,' Finished:',finished)
		m = file_match(ppl_dir,pdf_done)
		print(m, offset,i)

		if len(m.files) > offset: todo = m.files[offset:offset + max_nthreads]
		else: 
			finished = time.time()
			continue
		# check whether there is thread available and no pause is set
		if tm.thread_available('pdf') and not os.path.isfile('pause_pdf'):

			# check whether there is stuff to do
			print(todo, offset,i)

			finished = False
			for file_info in todo:
				pdf_name = pdf_dir + file_info.name + '.pdf'
				f = file_info.f
				print('ppl processing:',file_info.f,'saving to:',pdf_name)
				job_accepted = tm.add_thread('pdf',mlwp.make_pdf,(f,pdf_name,lexicon))
				if not job_accepted: print('pdf',file_info.f,'job NOT accepted')
				else: print('pdf',file_info.f,'job ACCEPTED')
				time.sleep(0.5)

		elif os.path.isfile('paused_pdf'): print('paused \n',tm,m)
		else: print('waiting... for available threads\n',tm,m)
		if finished and time.time() - finished > 7200: break
		time.sleep(3)

	with open('working_ponies','a') as fout: fout.write(socket.gethostname()+'\tpdf done\t'+time.strftime('%Y-%m-%d %H:%M:%S'))
	print(socket.gethostname() + 'done pdf', time.strftime('%Y-%m-%d %H:%M:%S'))
			
		


def handle_ppl(offset=0, max_nthreads=18,lt_dir = 'LEXICON_TXT/',ppl_done = 'DONE_PPL/',ppl_dir = 'PPL/', lm_name = 'cow_clean10.lm',remove = True, max_i = None):
	with open('working_ponies','a') as fout: fout.write(socket.gethostname() + '\tppl\t' +  time.strftime('%Y-%m-%d %H:%M:%S') + str(offset) + '\n')
	tm = mlwp.thread_master(max_nthreads)
	print(tm)
	i = 0
	finished = False

	while 1:
		if os.path.isfile('stop_ppl') or max_i ==i: break
		i += 1

		print('creating file matcher to see what needs to be done. Index:',i,' Finished:',finished)
		m = file_match(lt_dir,ppl_done)
		print(m, offset,i)

		if len(m.files) > offset: todo = m.files[offset:offset + max_nthreads]
		else: 
			finished = time.time()
			continue

		finished = False
		print(m, todo, offset,i)
		# check whether there is thread available and no pause is set
		if tm.thread_available('ppl') and not os.path.isfile('pause_ppl'):

			# check whether there is stuff to do
			for file_info in todo:
				ppl_name = ppl_dir + file_info.name + '.ppl'
				f = file_info.f
				print('ppl processing:',file_info.f,'saving to:',ppl_name)
				job_accepted = tm.add_thread('ppl',mlwp.make_ppl,(f,ppl_name,lm_name,remove))
				if not job_accepted: print('ppl',file_info.f,'job NOT accepted')
				else: 
					print('ppl',file_info.f,'job ACCEPTED')
					time.sleep(1)

		elif os.path.isfile('pause_ppl'): print('paused \n',tm,m)
		else: print('waiting... \n',tm,m)
		if finished and time.time() - finished > 7200: break
		time.sleep(3)

	with open('working_ponies','a') as fout: fout.write(socket.gethostname()+'\tpdf done\t'+time.strftime('%Y-%m-%d %H:%M:%S'))
	print(socket.gethostname() + 'done ppl', time.strftime('%Y-%m-%d %H:%M:%S'))



class file_match():
	def __init__(self,goal_dir,done_dir):
		self.t = time.time()
		self.string_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t))
		self.goal_dir = goal_dir
		self.done_dir = done_dir
		self.goal = file_timer(goal_dir)
		self.done = file_timer(done_dir)
		self.files = [f for f in self.goal.files if f not in self.done.files]

	def __repr__(self):
		return 'file_match\tnfiles not done: ' + str(len(self.files)) + '\t' + self.string_time

	def __str__(self):
		return self.__repr__()


class file_timer():
	def __init__(self,directory):
		self.directory = directory
		if self.directory[-1] != '/': self.directory += '/'
		self.fn = glob.glob(directory + '*')
		self.files = [file_info(f,i) for i,f in enumerate(self.fn)]
		self.files_sorted = self.files[:]
		self.files_sorted.sort()
		
	def __repr__(self):
		return 'file_timer\tnfiles: ' + str(len(self.fn))
		


class file_info():
	def __init__(self,f,i = -9):
		self.f = f
		self.index = i
		self.name = f.split('/')[-1].split('.')[0]
		self.filetype = f.split('.')[-1]
		self.info = os.stat(f)
		self.mtime = self.info.st_mtime
		self.string_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.mtime))
		self.mb = self.info.st_size / 10**6

	def __repr__(self):
		return 'file info: '+str(self.index)+' ' + self.filetype + '\tname: ' + self.name+'\thours ago: ' + str(round(self.hours_ago(),2)) + '\tsize: '+str(round(self.mb,2))

	def __str__(self):
		m = self.__repr__() +'\n'
		m += 'index:\t\t' + str(self.index) + '\n'
		m += 'time:\t\t' + self.string_time
		return m

	def __lt__(self,other):
		return self.mtime < other.mtime

	def __eq__(self,other):
		return self.name == other.name

	def seconds_ago(self):
		 return time.time() - self.mtime

	def minutes_ago(self):
		 return (time.time() - self.mtime) / 60

	def hours_ago(self):
		 return (time.time() - self.mtime) / 3600
