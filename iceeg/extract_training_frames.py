import experiment as e
import video_log as vl

for i in range(1,49):
	print('Loading participant:',i)
	p = e.Participant(i)
	p.add_all_sessions()
	for s in p.sessions:
		v = vl.video_log(s.log)
		if v.ok:
			for marker in v.marker2time.keys():
				if marker % 10 == 0:
					v.extract_frames(marker,5)
		else: print('skipping:',v.pp_id,v.exp_type,'no marker file/or log file')

		
