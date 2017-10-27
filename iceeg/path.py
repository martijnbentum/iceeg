import os
'''Defines paths to all data dependencies of the module.'''
if os.path.isdir('/Volumes/storage/'): volume = 'Volumes/storage/'
elif os.path.isdir('/Users/u050158/storage/'): volume = '/Users/u050158/storage/'
else: print('please add path to secondary data folder to the path.py file in the iceeg folder.')

data= volume + 'EEG_DATA_ifadv_cgn/'
if os.path.isdir(volume +'EEG'):eeg= volume + 'EEG/'
elif os.path.isdir('/Volumes/BAK4TB/EEG/'):eeg = '/Volumes/BAK4TB/EEG/'
elif os.path.isdir('/Volumes/BAK1TB/EEG/'): eeg = '/Volumes/BAK1TB/EEG/'
else: print('please add path to eeg files to path.py file in the iceeg folder')

log= data + 'LOG_FILES/'
marker= data + 'MARKER_FILES/'
cgn_annot = data + 'CGN_ANNOTATION/' 
ifadv_annot = data + 'IFADV_ANNOTATION/'
compo_pos = cgn_annot + 'POS_O/FROG_OUTPUT/'
compk_pos = cgn_annot + 'POS_K/FROG_OUTPUT/'
ifadv_pos = ifadv_annot + 'POS_IFADV/FROG_OUTPUT/'
cgn_awd = cgn_annot + 'TABLE_CGN2_AWD/'
cgn_ort = cgn_annot + 'TABLE_CGN2_ORT/'
ifadv_ort = ifadv_annot + 'ORT/'
ifadv_awd = ifadv_annot + 'AWD/'

blinks = data + 'BLINKS/'
plot_blinks = data + 'PLOT_BLINKS/'
artifacts = data + 'ARTIFACTS/'
data_stats = data + 'DATA_STATS/'

video_log = data + 'VIDEO_LOG_EEG/'
video_data = '/Volumes/BAK4TB/EEG_VIDEO/'
video_frames= '/Volumes/BAK4TB/EEG_VIDEO_FRAMES/'



