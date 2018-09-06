import glob
# import mail
import os
import path
import paramiko
import time

class report_sender():
	'''Object to monitor model training output and computer statistics.
	sends a report of each new evaluation of a model
	monitors cpu and ram to check whether pc is still training (pc ok works, does not seem to
	send a notification when it fails.
	'''
	def __init__(self,check_interval = 1800,recipients = 'bentummartijn@gmail.com',purge_file = True):
		'''create send report object to monitor model training and pc statistics.

		target_dir 		directory to monitor
		check_interval 	time between pc monitoring and report (model evaluation) monitoring
		recipients 		email adresses to send reports to
		'''
		self.check_interval = check_interval
		self.start = time.time()
		self.recipients = recipients
		self.last_computer_report = 0
		self.last_monitor = 0
		self.names = 'blossomforth,cheerilee,fancypants,featherweight,fluttershy,pipsqueak,rarity,scootaloo,thunderlane,twist'.split(',')
		self.ponies = {}
		for name in self.names:
			self.ponies[name] = pony(name,purge_file = purge_file)
		
	def run(self):
		'''Periodically check for new reports and computer status.'''
		while 1:
			# if time.time() - self.last_computer_report > self.computer_report_job:
				# self.send_computer_report()
			if time.time() - self.last_monitor> self.check_interval:
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

	def get_command(self,command = 'free -g',info_type = 'ram'):
		host_names = [n + '.science.ru.nl' for n in self.names]
		user = 'mbentum'
		ponies = {}

		for i,hn in enumerate(host_names):
			print(user,hn)
			ssh = paramiko.SSHClient()
			ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
			ssh.connect(hn,username=user,key_filename='/Users/u050158/.ssh/id_rsa.pub')
			stdin,stdout,stderr = ssh.exec_command(command)
			info = stdout.read().decode()
			print(info)
			ponies[self.names[i]] = info
			ssh.close() 

		setattr(self,info_type,ponies)

	def get_info(self):
		host_names = [n + '.science.ru.nl' for n in self.names]
		user = 'mbentum'
		self.ram = {}
		self.cpu= {}

		for i,hn in enumerate(host_names):
			print(user,hn)
			ssh = paramiko.SSHClient()
			ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
			ssh.connect(hn,username=user,key_filename='/Users/u050158/.ssh/id_rsa.pub')

			stdin,stdout,stderr = ssh.exec_command('free -g')
			info = stdout.read().decode()
			print(info)
			self.ram[self.names[i]] = info

			c = 'top -b -n1 | grep "Cpu(s)" | awk \'{print $2 + $4}\''
			stdin,stdout,stderr = ssh.exec_command(c)
			info = stdout.read().decode()
			print(info)
			self.cpu[self.names[i]] = info
			ssh.close() 


	def monitor_computer(self):
		self.last_monitor = time.time()
		self.get_info()
		for name in self.names:
			total_mem, used_mem, free_mem = ram2info(self.ram[name])
			cpu =  cpu2info(self.cpu[name])
			self.ponies[name].set_info(total_mem, free_mem, used_mem, cpu)
		
		

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


def ram2info(ram):
	ram = [item for item in ram.split(' ') if item]
	try:
		total_mem, used_mem, free_mem = map(int,ram[6:9])
	except:
		total_mem, used_mem, free_mem = 0, 0, 0
	return total_mem, used_mem, free_mem
		
def cpu2info(cpu):
	try:
		cpu = float(cpu) 
	except:
		cpu = 0
	return cpu



class pony:
	def __init__(self,name,total_mem=0,free_mem=0,used_mem=0,cpu=0, filename = '',purge_file = False):
		self.name = name
		if filename == '': self.filename = name + '.txt'
		self.hist_free_mem, self.hist_used_mem , self.hist_perc_mem_free= [], [], []
		self.hist_cpu, self.hist_time, = [],[]
		self.set_info(total_mem,free_mem,used_mem,cpu)
		self.write_info(True)
		
		
	def set_info(self,total_mem = 0,free_mem = 0,used_mem = 0,cpu = 0):
		self.total_mem = total_mem

		self.free_mem = free_mem
		self.hist_free_mem.append(free_mem)

		self.used_mem = used_mem
		self.hist_used_mem.append(used_mem)

		self.cpu = cpu
		self.hist_cpu.append(self.cpu)

		try: self.perc_mem_free = int(free_mem) / int(total_mem)
		except: self.perc_mem_free = 0
			
		self.hist_perc_mem_free.append(self.perc_mem_free)
		self.time = time.time()
		self.hist_time.append(self.time)
		self.write_info()


	def string_info(self):
		return '\t'.join(list(map(str,[self.free_mem,self.used_mem,self.cpu,time.strftime("%Y-%m-%d %H:%M", time.localtime(self.time))]))) + '\n'


	def write_info(self,purge = False):
		if purge: fout = open(self.filename,'w')
		else: fout = open(path.data + self.filename,'a')
		fout.write(self.string_info())
		fout.close()

	

