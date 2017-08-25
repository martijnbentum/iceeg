import experiment as e
import video_log as vl

output = []
ld = []

for i in range(1,49):
	output.append('Loading participant: '+str(i))
	p = e.Participant(i)
	p.add_all_sessions()
	for s in p.sessions:
		v = vl.video_log(s.log)
		if v.marker_exist:
			output.extend(v.__str__().split('\n'))
			if hasattr(v,'lengthdelta'): 
				ld.append(str(v.pp_id) +'\t'+v.exp_type+'\t'+str(v.lengthdelta)+'\t'+v.log_duration_str)
			else: 
				ld.append ('skipping: '+str(v.pp_id)+' '+v.exp_type+' no-length-delta')
		else: output.append('skipping: '+str(v.pp_id)+' '+v.exp_type+' no marker file')

print(output)
print(' ')
print('-'*90)
print(' ')
print(ld)
		
		
