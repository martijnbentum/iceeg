import experiment as e
import video_log as vl
import os

commands = []

for i in range(1,49):
	# print('Loading participant:',i)
	p = e.Participant(i)
	p.add_all_sessions()
	for s in p.sessions:
		v = vl.video_log(s.log)
		if v.ok:
			fps = str(1 / v.adj_framelength)
			ifile = v.video_info.video_name
			ofile = ifile.replace('avi','mkv')
			c = 'ffmpeg -r ' + fps + ' -i ' + ifile + ' -f matroska -vcodec libx264 ' + ofile
			commands.append(c)
			# os.system(c)
		else: print('skipping:',v.pp_id,v.exp_type,'no marker file/or log file')

fout = open('mkv_commands.txt','w')
fout.write('\n'.join(commands))
fout.close()
		
