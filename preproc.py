# import filter_butterworth as bw
import matplotlib.pyplot as plt
import mne

plt.ion()

def load_block(b):
	'''Load eeg data corresponding to 1 block.

	should think about filtering and edge effects
	'''
	vhdr_fn = b.vmrk.vmrk_fn.replace('.vmrk','.vhdr')
	raw = load_eeg(vhdr_fn = vhdr_fn)

	start_sec = b.st_sample / 1000
	end_sec = b.et_sample / 1000

	raw.crop(tmin = start_sec, tmax = end_sec)
	raw.load_data()
	
	raw = rereference(raw)
	raw = filter_iir(raw)
	raw = make_eog_diff(raw)

	m = mne.channels.read_montage('easycap-M1')
	raw.set_montage(m)
	return raw
	
	

def load_eeg(pp_id = 'pp001',exp = 'ifadv',path = '../EEG/',vhdr_fn = 'None', preload = False):
	'''Load eeg data.

	vhdr_fn 	file is loaded, if not provided, otherwise vhdr_fn is made 
	from pp_id and exp
	'''
	if vhdr_fn != None:
		print('Using filename:',vhdr_fn,' pp_id and exp is ignored' )
	else:
		vhdr_fn = path + pp_id + '_' + exp + '.vhdr'
	print('loading: ',vhdr_fn)
	raw = mne.io.read_raw_brainvision(vhdr_fn,preload =preload)
	# if you specify eog channels these channels will not be filtered
	return raw

def make_eog_diff(raw):
	'''Make VEOG and HEOG channels (difference wave between up low / left right.
	
	should check whether to subtract left from right or reverse.
	'''
	raw = make_diff_wav(raw,'Fp1_EOG_V_high','Oz_EOG_V_low','VEOG',False)
	raw = make_diff_wav(raw,'FT9_EOG_H_left','FT10','HEOG',False)
	raw.set_channel_types({'VEOG':'eog','HEOG':'eog'})
	return raw


def make_diff_wav(raw,ch_name1,ch_name2,new_ch_name,copy = True):
	'''Make difference wave from two channels.

	Subtract ch2 from ch1 and store it in ch1 and name it new_ch_name
	If copy is true only return difference wave 
	Else return difference wav and delete ch1 and ch2.
	'''
	if copy: output = raw.copy()
	else: output = raw
	i1 = output.ch_names.index(ch_name1)
	i2 = output.ch_names.index(ch_name2)
	output[i1][0][0] = output[i1][0][0] - output[i2][0][0]
	output.rename_channels({ch_name1:new_ch_name})
	if copy: output.pick_channels([new_ch_name])
	else: output.drop_channels([ch_name2])
	return output

 
def rereference(raw):
	'''Rereference data to linked left and right mastoid electrodes.'''

	print('add empty reference channel LM and use reference function to use \
	average of LM and TP10_RM (average of mastoids)')

	# adds an empty channel (all zeros, for the left mastiod
	r = mne.add_reference_channels(raw,'LM',True)
	# rereference to the mean of the LM and RM
	r.set_eeg_reference(ref_channels = ['LM','TP10_RM'])
	# I visually (plot) checked that the RM value is half of what is was before
	#to be able to set montage (electrode locations) reference electrodes
	# (and eog) should not be of type eeg
	r.set_channel_types({'TP10_RM':'misc','LM':'misc'})
	return r
	

def filter_iir(raw,order = 5,freq = [0.05,30],sf = 1000,pass_type = 'bandpass'):
	'''Filter data with butterworth filter.

	# MNE default =  iir_params is None and method="iir", 
		4th order Butterworth will be used
 
	- 'iir' will use IIR forward-backward filtering (via filtfilt).
	- This function will use bandpass filter butterworth order 5 0.05 - 30 Hz
	- Filtering is done in place
	'''
	iir_params = dict(order=order, ftype='butter',output = 'sos')
	iir_params = mne.filter.construct_iir_filter(iir_params, freq,None,sf, \
		pass_type, return_copy = False)
	print('creating IIR butterworth filter with following params:\n',iir_params)
	print('frequency cut off:','\t'.join(map(str,freq)))
	print('sample frequency:',sf)
	print('filter pass_type:',pass_type)
 
	raw.filter(iir_params =iir_params,l_freq= freq[0],h_freq=freq[1],method = 'iir')
	return raw 


def plot(df):
	plt.plot(df.Fz)
	plt.show()

def plot_eog(df):
	plt.plot(df.Fp1_EOG_V_high - df.Oz_EOG_V_low)
	plt.legend('vertical eog difference')
	plt.show()

def plot_eogz(df):
	fp1z, ozz, dif, difz, combz = dif_zscore(df)
	plt.plot(fp1z)
	plt.plot(ozz)
	plt.plot(dif)
	plt.plot(difz)
	plt.plot(combz)
	plt.legend(('fp1z','ozz','dif','difz','combz'))
	plt.show()

def df_ch2zscore(ch):
	return (ch - ch.mean()) / ch.std()

def dif_zscore(df):
	fp1z = df_ch2zscore(df.Fp1_EOG_V_high)
	ozz = df_ch2zscore(df.Oz_EOG_V_low)
	dif = df_ch2zscore(df.Fp1_EOG_V_high - df.Oz_EOG_V_low)
	difz = fp1z - ozz
	combz = (fp1z - ozz) + dif + difz
	return fp1z, ozz, dif, difz, combz


def preproc(raw = None,pp = None,pp_id = None, exp = None,tmin_sec = 0.0,tmax_sec = 5900.0):
	if raw == None: 
		if pp:
			pass
		if pp_id == None:
			raw = load_eeg()
		elif exp == None:
			raw = load_eeg(pp_id)
		else:
			raw = load_eeg(pp_id,exp)
		raw.crop(tmin = tmin_sec,tmax=tmax_sec)
	
	raw = rereference(raw)
		
	df_raw = raw.to_data_frame()
	diir = filter_iir(raw.copy())
	df_iir = diir.to_data_frame()

	plt.cla()
	plot(df_raw)
	plot(df_iir)
	plt.legend(['raw','iir'])
	plt.grid()
	return raw,diir,df_iir




