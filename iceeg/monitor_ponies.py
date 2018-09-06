import paramiko
import time


def command_all(command = 'echo $HOSTNAME'):
	host_names = [n + '.science.ru.nl' for n in 'blossomforth,cheerilee,fancypants,featherweight,fluttershy,pipsqueak,rarity,scootaloo,thunderlane,twist'.split(',')]

	user = 'mbentum'

	for hn in host_names:
		print(user,hn)
		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh.connect(hn,username=user,key_filename='/Users/u050158/.ssh/id_rsa.pub')
		stdin,stdout,stderr = ssh.exec_command(command)
		'\n'.join(stdout.readlines())
		ssh.close() 
'''                                                                                                     
hostname = 'my hostname or IP'                                                                          
myuser   = 'the user to ssh connect'                                                                    
mySSHK   = ‘.ssh/id_rsa'                                                                                
sshcon   = paramiko.SSHClient()  # will create the object                                               
sshcon.set_missing_host_key_policy(paramiko.AutoAddPolicy())# no known_hosts error                      
sshcon.connect(hostname, username=myuser, key_filename=mySSHK)  
'''
