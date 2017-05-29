import filter_butterworth as bw
import matplotlib.pyplot as plt
import mne

plt.ion()


def load_eeg(pp_id = 'pp001',exp = 'ifadv',path = 'EEG/',fn = 'None'):
	if fn != None:
		print( 999)
		fn = path + pp_id + '_' + exp + '.vhdr'
	print('loading: ',fn)
	raw = mne.io.read_raw_brainvision(fn,preload =True)
	return raw

 
def rereference(raw):
	print('add empty reference channel LM and use reference function to use \
	average of LM and TP10_RM (average of mastoids)')

	# adds an empty channel (all zeros, for the left mastiod
	r = mne.add_reference_channels(raw,'LM',True)
	# rereference to the mean of the LM and RM
	r.set_eeg_reference(ref_channels = ['LM','TP10_RM'])
	# I visually (plot) checked that the RM value is half of what is was before
	return r
	

def filter_iir(raw,order = 5,freq = [0.05,30],sf = 1000,pass_type = 'bandpass'):
	# This is the filter i am currently using
	# MNE default =  iir_params is None and method="iir", 4th order Butterworth will be used
	# 'iir' will use IIR forward-backward filtering (via filtfilt).
	# This function will use bandpass filter butterworth order 5 0.05 - 30 Hz
	iir_params = dict(order=order, ftype='butter',output = 'sos')
	iir_params = mne.filter.construct_iir_filter(iir_params, freq,None,sf, \
		pass_type, return_copy = False)
	print('creating IIR butterworth filter with following params:\n',iir_params)
	print('frequency cut off:','\t'.join(map(str,freq)))
	print('sample frequency:',sf)
	print('filter pass_type:',pass_type)
 
	raw.filter(iir_params =iir_params,l_freq= freq[0],h_freq=freq[1],method = 'iir')
	return raw # filtering is done in place


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
	plt.legend(['raw','fir','iir'])
	plt.grid()
	return raw,diir,df_iir




