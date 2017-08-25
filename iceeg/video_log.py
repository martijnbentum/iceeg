import calendar
import glob
import os
import pandas as pd
import path

class video_log:
	'''Aggregate information about start/end times in the videos.
	synchronize camera and eeg experiment clocks - Work in progress
	Check for dropped frames and adj framelength to spread out misalignment 
	~ 9 seconds in 1.5 hours - Work in progress

	Extracts frames from videos (4.7 gb .avi)

	Should maybe export videos for each block to speed up subsequent work
	'''
	
	def __init__(self, log):
		'''Aggregate information about start/end times of events in the videos.
		
		Keywords:
		log 	log object created with log.py
		'''
		self.log = log
		self.pp_id = log.pp_id
		self.exp_type = log.exp_type
		self.start = log.start_exp
		self.eeg_start_str = log.start_exp
		self.end = log.end_exp
		self.ok = False
		self.init()


	def __str__(self):
		if self.videolog_fn == None: return 'No video log file found.'
		m = 'videolog_fn:\t\t' + self.videolog_fn + '\n'
		m += 'eeg_start_str:\t\t' + str(self.eeg_start_str) + '\n'
		m += 'log_start_str:\t\t' + str(self.log_start_str) + '\n'
		m += 'timedelta:\t\t' + str(self.timedelta) + '\n'
		m += 'marker_fn:\t\t' + str(self.marker_fn) + '\n'
		m += 'marker_exist:\t\t' + str(self.marker_exist) + '\n'
		m += 'log_start:\t\t' + str(self.log_start_time) + '\n'
		m += 'log_end:\t\t' + str(self.log_end_time) + '\n'
		if self.ok
			m += 'log_duration_str:\t' + str(self.log_duration_str) + '\n'
			m += 'log_duration_sec:\t' + str(self.log_duration_sec) + '\n'
			m += 'lengthdelta:\t\t' + str(self.lengthdelta) + '\n'
			m += 'frames_dropped:\t\t' + str(self.frames_dropped) + '\n'
			m += 'adj:\t\t\t' + str(self.adj) + '\n'
			m += 'adj_framelength:\t' + str(self.adj_framelength) + '\n'
		m += 'VIDEO_INFO:\n'
		if self.video_info: m += self.video_info.__str__()
		else: m += "no info"
		return m


	def init(self):
		'''Create videolog object with information about start and end times.'''
		self.ctime = self.start.ctime()
		self.marker_fn = path.marker + self.ctime.split(' ')[1] + '_' + str(self.start.day)+'_2017.txt'
		self.marker_exist = os.path.isfile(self.marker_fn)
		self.find_videolog()
		if self.marker_exist: self.load_marker()
		else: return 0
		if self.videolog_fn == None: return 0
		else: 
			self.log_start_str = self.videolog_fn.split('/')[-1].strip('pp_log_').strip('.log')
			self.videolog = open(self.videolog_fn).read().split('\t')
			self.log_start_time = float(self.videolog[0])
			self.log_end_time = float(self.videolog[1])
			self.log_duration_sec = float(self.videolog[2])
			self.log_duration_str = self.videolog[3]
			self.video_info = video_info(self.videolog_fn)
			self.lengthdelta = self.log_duration_sec - self.video_info.duration_sec
			self.frames_dropped = int(self.lengthdelta / self.video_info.frame_length)
			self.adj = self.lengthdelta / self.video_info.nframes
			self.adj_framelength = self.video_info.frame_length + self.adj
			self.make_marker2videotime()
			self.ok = True

	def make_marker2videotime(self):
		'''Create dictionary that transelates eeg markers to videotime.'''
		if not self.marker_exist:
			print('no marker file present')
			return 0
		self.marker2videotime = {}
		for marker in self.marker2time.keys():
			self.marker2videotime[marker] = self.marker2time[marker] - self.log_start_time

	def videotime2framenumber(self,videotime):
		'''Translate videotime to framenumber, adjusting for dropped frames.
		The difference in length between the video and recording time is spread
		out over the frames so we use the adjusted framelength.
		'''
		return int(videotime / self.adj_framelength)

	def blocktime2framenumber(self,marker,offset=0):
		'''Transelate marker and offset to a specific framenumber adjusting for
		dropped frames.
		'''
		if not self.marker_exist:
			print('no marker file present')
			return 0
		return self.videotime2framenumber(self.marker2videotime[marker] + offset)

	def blocktime2adj_videotime(self,marker,offset = 0):
		'''Transelate marker and offset to adjusted videotime, this accounts
		for dropped frames by using the framenumber calculated with adjusted
		framelength and multiplying it with video framelength.
		The adjusted videotime is before the blocktime (the video is shorter
		because of the dropped frames, typically 9 sec)
		'''
		framenumber = self.blocktime2framenumber(marker,offset)
		return framenumber * self.video_info.frame_length

	def extract_frames(self,marker,nframes = 1,interval= None,ft = '.bmp'):
		'''Extract 1 or more frames from a block specified by marker.
		nframes can be all to extract all frames from block
		interval specifies the time between frames in seconds if it is unspecified
		it is calculated based on the number of frames and the duration of the block
		ft specifies filetype
		'''
		
		frame_number = self.blocktime2framenumber(marker)
		start = self.blocktime2adj_videotime(marker)
		start_str = self.video_info.sec2vid(start)
		duration = self.marker2videotime[marker+1] - self.marker2videotime[marker]  
		name = 'pp'+str(self.pp_id)+'_'+self.exp_type+'_'+str(marker)
		directory = path.video_frames + name + '/'
		if not os.path.isdir(directory): os.mkdir(directory)
		if nframes == 'all': 
			nframes = int(duration / self.adj_framelength)
			interval = self.video_info.frame_length
		if interval == None and nframes > 1:
			interval = int(duration/nframes)
			if interval > 15: interval = 15

		c = 'ffmpeg -ss '+start_str+ ' -i '+self.video_info.video_name
		if nframes == 1: 
			of= directory+name+'_'+str(frame_number)+ft
			if os.path.isfile(of):
				print('frames already extracted')
				return 0
			c += ' -vframes 1 '+ output_filename
		else:
			of= directory+name+'_'+str(frame_number)+'_'+str(interval)+'_'+'%04d'+ft
			print(of)
			print(of.replace('%04d','0001'))
			if os.path.isfile(of.replace('%04d','0001')):
				print('frames already extracted')
				return 0
			c += ' -vf fps=1/'+str(interval)+' -vframes '+str(nframes)+' '+of
		print('Extracting',nframes,'frames')
		print(c)
		os.system(c)
		

	def load_marker(self):
		'''Load file with marker number and corresponding epoch time of camera
		clock. This clock is not synchronized with eeg experiment clock'''
		temp = [[int(line.split('\t')[0]), float(line.split('\t')[1])] for line in open(self.marker_fn).read().split('\n') if line]
		start,end = 0,0
		for i,line in enumerate(temp):
			if start > 0 and end == 0 and line[0] == 255 and i > start + 3:
				end = i
			if start == 0 and line[0] == 255 and temp[i+3][0] == 255 and temp[i+1][0] == self.pp_id: 
				start = i + 4
		if end == 0: end = len(temp)
		self.marker = temp[start:end]
		self.marker2time = dict(self.marker)


	def find_videolog(self):
		'''Find the the file that is corresponds to a experiment log.
		The videolog contain start and end times of the video recording.
		The have no identifier so only the timestamp can be used to identify them
		'''
		self.fn_log = glob.glob(path.video_log + '*.log')
		self.possible_f = []
		for f in self.fn_log:
			t = self.f_log2pd_time(f)
			if self.start > t: diff = self.start - t
			else: diff = t - self.start
			if abs(diff.seconds) < 3600 and diff.days == 0:
				self.possible_f.append([f,t,diff])
			# else: print(diff,t,self.start>t)
		if len(self.possible_f) == 0: self.videolog_fn = None
		elif len(self.possible_f) == 1: 
			self.videolog_fn = self.possible_f[0][0]
			self.timedelta = self.possible_f[0][2]
		else: self.find_closest_videolog()

	def find_closest_videolog(self):
		'''Find the videolog that is closest in time to the eeg log.'''
		best_index = 0
		best_diff = self.possible_f[0][-1]
		for i,line in enumerate(self.possible_f):
			if line[-1] < best_diff:
				best_index = i
				best_diff = line[-1]
		self.videolog_fn = self.possible_f.pop(best_index)
		self.timedelta = self.videolog_fn[2]
		self.videolog_fn = self.videolog_fn[0]
			
	def read_videolog(self):
		'''Extract start end and duration from the videolog.'''
		self.videolog_text = open(self.videolog_fn).read().split('\t')
		self.videolog_epoch_start = float(self.videolog_text[0])
		self.videolog_epoch_end = float(self.videolog_text[1])
		self.videolog_duration_sec = float(self.videolog_text[2])
		self.videolog_duration_text = self.videolog_text[3]
		

	def f_log2pd_time(self,f):
		'''Convert videolog timestamp to panda timestamp.'''
		t = f.split('/')[-1].lstrip('pp_log_').split('.')[0].split('-')
		y,mo,d,h,mi,se = map(int,t)
		return pd.Timestamp(year = y,month = mo, day = d, hour = h, minute = mi, second = se)



class video_info:
	'''Extract information about a videofile, based on ffmpeg outputfiles.'''

	def __init__(self,name):
		'''Create a information object about a videofile

		name 	name of the videolog, is used to find the ffmpeg information file.
		'''
		self.name = name.strip('.log').replace('log','video') + '.frames'
		self.video_name = path.video_data + self.name.split('/')[-1].replace('frames','avi')
		self.info = open(self.name).read().split('\n')
		temp = self.info[-3].split(' ')
		self.nframes = int(temp[0].split('=')[-1])
		self.duration_str = temp[4].split('=')[-1]
		self.duration_sec = self.vid2sec()
		self.bitrate = temp[-2].split('=')[-1]
		temp = self.info[14].split(',')
		self.codec = temp[0].split(' ')[-4]
		self.window_size = temp[-5].split(' ')[1]
		self.width = int(self.window_size.split('x')[0])
		self.height = int(self.window_size.split('x')[1])
		self.fps = int(temp[-4].split(' ')[1])
		self.frame_length = 1 / self.fps

	def __str__(self):
		m = 'name\t\t' + self.name +'\n'
		m += 'video_name\t' + self.video_name+'\n'
		m += 'duration str\t' + self.duration_str +'\n'
		m += 'duration sec\t' + str(self.duration_sec)+'\n'
		m += 'bitrate\t\t' + str(self.bitrate)+'\n'
		m += 'codec\t\t' + self.codec+'\n'
		m += 'window_size\t' + str(self.window_size)+'\n'
		m += 'width\t\t' + str(self.width)+'\n'
		m += 'height\t\t' + str(self.height)+'\n'
		m += 'fps\t\t' + str(self.fps)+'\n'
		m += 'frame_length\t' + str(self.frame_length)+'\n'
		m += 'nframes\t\t' + str(int(self.nframes))+'\n'
		return m

	def vid2sec(self,vid = None):
		'''Transelate video format time 00:00:00.00 to seconds.'''
		if vid == None: vid = self.duration_str
		if vid == 'N/A': 
			return None
		h,m,s = vid.split(':')
		return float(h) * 3600 + float(m) * 60 + float(s)

	def sec2vid(self,sec = 0 ):
		'''Transelate seconds to video format time 00:00:00.00.'''
		h = str(int(sec / 3600))
		sec = sec % 3600
		m = str(int(sec / 60))
		sec = str(sec % 60)
		return ':'.join([h,m,sec])
