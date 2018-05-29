from apscheduler.schedulers.background import BackgroundScheduler as bs
import glob
import mail
import os
import path
import plot_roc
import sched
import time

class report_sender():
	'''Object to monitor model training output and computer statistics.
	sends a report of each new evaluation of a model
	monitors cpu and ram to check whether pc is still training (pc ok works, does not seem to
	send a notification when it fails.
	'''
	def __init__(self,target_dir = '/Volumes/Bigstorage/BAK/MODEL_CHANNEL/',check_interval = 60,recipients = 'bentummartijn@gmail.com'):
		'''create send report object to monitor model training and pc statistics.

		target_dir 		directory to monitor
		check_interval 	time between pc monitoring and report (model evaluation) monitoring
		recipients 		email adresses to send reports to
		'''
		self.target_dir = target_dir
		self.check_interval = check_interval
		self.fn = []
		self.start = time.time()
		self.recipients = recipients
		self.reports = []
		self.mem_to_low = False
		self.cpu_to_low = False
		self.min_mem = 8000
		self.min_cpu = 50
		self.last_computer_report = 0
		self.cpu = []
		self.mem = []
		self.first_run = True
		self.short_monitor_interval = False
		self.sched = bs()
		self.sched.start()
		self.report_job = check_interval + 1
		self.monitor_job = check_interval
		self.computer_report_job = 7200
		self.check_new_reports()
		self.send_computer_report()
		
	def run(self):
		'''Periodically check for new reports and computer status.'''
		while 1:
			if time.time() - self.last_computer_report > self.computer_report_job:
				self.send_computer_report()
			if time.time() - self.last_report> self.report_job:
				self.check_new_reports()
			if time.time() - self.last_monitor> self.monitor_job:
				self.monitor_computer()
			time.sleep(30)


	def add_recipient(self,r = ''):
		'''Add an email adress to the recipient list.'''
		if r != '' and '@' in r:
			self.recipients += ',' + r


	def remove_recipient(self,r = '',index= 'NA'):
		'''Remove an email from the recipient list.'''
		if index != 'NA' and type(index) == int:
			recipients = self.recipients.split(',')
			removed_recipient = recipients.pop(i)
			self.recipients = ','.join(recipients)
		if r != '' and '@' in r:
			try:
				i = self.recipients.split(',').index(r)
				removed_recipient = recipients.pop(i)
				self.recipients = ','.join(recipients)
			except: pass
		print('remove recipient:', removed_recipient)
		print('remaining recipients:', self.recipients)



	def check_new_reports(self):
		'''Check whether a new model evaluation report is present in the target directory.'''
		self.last_report = time.time()
		fn = glob.glob(self.target_dir + '*report*')
		last_10_acc = '\n\nLast 10 accuracy test 50/50, artifact/clean:\n\n'
		last_10_acc += '\n'.join(open(path.data +'test_output.txt').read().split('\n')[::-1][:11])
		for f in fn:
			if f not in self.fn:
				self.fn.append(f)
				if not self.first_run:
					cmc = plot_roc.cm_collection()
					cmc.plot(plot_type = 'roc',save = True)
					cmc.plot(plot_type = 'mcc',save = True)
					cmc.plot(plot_type = 'pr',save = True)
					cmc.plot(plot_type = 'f',save = True)
					self.reports.append(open(f).read())
					subject = ' '.join(f.split('_')[2:-2])
					message = self.reports[-1] + '\n\n' + last_10_acc + '\n\n'
					mail.mail(message,subject = subject, to = self.recipients,attach = 'roc.png,mcc.png,pr.png,f.png')
					time.sleep(10)
		self.first_run = False


	def monitor_computer(self):
		'''Check wheter cpu and ram is working as expected (does not seem to trigger if its not the case.'''
		self.last_monitor= time.time()
		memory = os.popen('top -l 1 -s 0 | grep PhysMem').read()
		cpu = os.popen('top -l 1 -s 0 | grep CPU\ usage').read()
		mem = memory.split(' ')[1]
		if mem[-1] == 'G': mem = int(mem[:-1]) * 1000
		elif mem[-1] == 'M': mem = int(mem[:-1]) 
		else: print('could not parse:',mem)
		
		cpu= cpu.split(' ')[2]
		if cpu[-1] == '%': cpu = float(cpu[:-1])
		else: print('could not parse:',cpu)
		clock = time.strftime("%H:%M:%S\t%b-%d-%Y", time.localtime(time.time()))
		self.cpu.append(str(cpu) + '\t' + clock)
		self.mem.append(str(mem) + '\t' + clock)

		# check cpu
		if cpu < self.cpu_to_low:
			if self.cpu_to_low:  
				print('cpu to low')
				if time.time() - self.start_cpu_to_low > 900 and not self.cpu_send: 
					mail.mail('\n'.join(self.cpu),subject = 'cpu is to low')
					self.cpu_send = True
			else:
				self.cpu_to_low = True
				self.start_cpu_to_low = time.time() 
				self.cpu_send = False
		else:
			self.cpu_to_low = False
			self.cpu_send = False
			
		# check memory
		if mem < self.mem_to_low:
			if self.mem_to_low:  
				print('mem to low')
				if time.time() - self.start_mem_to_low > 900 and not self.mem_send: 
					mail.mail('\n'.join(self.mem),subject = 'mem is to low')
					self.mem_send = True
			else:
				self.mem_to_low = True
				self.start_mem_to_low = time.time() 
				self.mem_send = False
		else:
			self.mem_to_low = False
			self.mem_send = False

		if self.cpu_to_low or self.mem_to_low:
			if not self.short_monitor_interval:
				self.short_monitor_interval = True
				self.monitor_job = 30
		elif self.short_monitor_interval:
			self.sort_monitor_interval = False
			self.monitor_job = self.interval
			
		

	def send_computer_report(self):
		'''Send a report with cpu and ram usage.'''
		self.last_computer_report = time.time()
		if self.cpu_to_low or self.mem_to_low:
			return 0
		self.monitor_computer()
		message =  '\n\nMEMORY\n------\n\n' +'\n'.join(self.mem[::-1][:30]) + '\n\nCPU\n---\n\n' + '\n'.join(self.cpu[::-1][:30])
		subject = 'PC OK, MEM: ' + self.mem[-1].split('\t')[0] + ' CPU: ' + self.cpu[-1].split('\t')[0]
		mail.mail(message_text=subject + '\n\n' +message,subject =subject)

		self.last_computer_report = time.time()



'''

		# self.report_job = self.sched.add_job(self.check_new_reports,'interval',minutes = 20,misfire_grace_time = 60)
		# self.monitor_job = self.sched.add_job(self.monitor_computer,'interval',minutes = 20,misfire_grace_time =60)
		# self.computer_report_job= self.sched.add_job(self.send_computer_report,'interval',hours = 2,misfire_grace_time = 120,coalesce = True)

'''
