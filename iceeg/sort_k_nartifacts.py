import glob
import path
import xml_handler

'''
Create a text file with all block names of an experiment, sorted by the number of artifacts
with or without artifact information
The block name list can be used by cnn_artifact_corrector to help annotation.
'''


def read_files(exp):
	'''Read all uncorrected automatic epoch artifact files of a experiment.'''
	return glob.glob(path.artifact_cnn_xml + '*exp-'+exp+'*.xml')

def read_ch_files(exp):
	'''Read all uncorrected automatic channel artifact files of a experiment.'''
	return glob.glob(path.artifact_ch_cnn_xml + '*exp-'+exp+'*.xml')

def get_name(f):
	'''Get block name from xml file (remove cnn model name).'''
	return f.split('part-90_')[-1].split('.')[0]

def get_artifacts(xml,filetype = 'artifacts',remove_ch = ['Fp2']):
	'''Read in the xml file and create a list of channel or epoch artifacts.'''
	print('retreiving:',filetype)
	if filetype == 'artifacts': 
		return [be for be in xml.bad_epochs if be.annotation == 'artifact']
	elif filetype == 'channels':
		return [bc for bc in xml.bad_channels if bc.annotation == 'artifact' and bc.channel not in remove_ch]
	else: raise ValueError('filetype not recognized: artifacts or channels',filetype)
	

def load_xml(f,filetype = 'artifacts'):
	'''Read the xml file.
	filetype 		channel or artifacts (epochs)
	'''
	xml = xml_handler.xml_handler(filename=f)
	xml.load_xml()
	if filetype == 'artifacts': xml.xml2bad_epochs()
	elif filetype == 'channels': xml.xml2bad_channels()
	else: raise ValueError('filetype not recognized: artifacts or channels',filetype)
	return xml

def write(line,filename):
	'''write line with block name and optionally artifact information.'''
	with open(path.data + filename,'a') as fout:
		fout.write(line + '\n')

def load(filename):
	'''Load a file with block names of an experiment (to add lines to).'''
	try:
		d = [line.split('\t') for line in open(path.data + filename).read().split('\n') if line]
		names = [line[0] for line in d]
		return d, names
	except: return False, False


def nartifacts(exp = 'k',overwrite = True,filetype = 'artifacts',remove_ch = ['Fp2']):
	'''Create a file of all block in an experiment ordered on the number of artifacts.
	this will help annotation.
	exp 		experiment type, k o ifadv
	overwite 	whether to create a new file or add to an old one
	filetype 	whether to create a file for epoch(artifact) or channel
	remove_ch 	channels to ommit by default
	'''
	filename = 'nartifacts_' +filetype+'_'+ exp
	if overwrite: 
		open(path.data + filename,'w').close()
	if filetype == 'artifacts': fn = read_files(exp)
	elif filetype == 'channels': fn = read_ch_files(exp)
	else: raise ValueError('filetype not recognized: artifacts or channels',filetype)
	d,names = load(filename)
	output = []
	for i,f in enumerate(fn):
		name = get_name(f)
		print(name,i,len(fn))
		# if not overwrite previous info and name is already in file continue to next name
		if not overwrite and names and name in names: continue
		xml = load_xml(f,filetype=filetype)
		artifacts = get_artifacts(xml,filetype = filetype,remove_ch= remove_ch)
		nartifacts = len(artifacts)
		duration = sum([a.duration for a in artifacts])
		line = '\t'.join([name,str(nartifacts),str(duration)])
		write(line,filename)
		output.append([name,nartifacts,duration])
	return output

def sort_names(sort_type = 'duration',exp = 'k',filetype = 'artifacts',overwrite = True, remove_ch = ['Fp2']):
	'''Sort the names based on the number of artifacts.'''
	output = nartifacts(exp = exp, overwrite = overwrite,filetype = filetype,remove_ch = remove_ch)
	if sort_type == 'duration':
		output.sort(key = lambda x: x[2])
	elif sort_type == 'nartifacts':
		output.sort(key = lambda x: x[1])
	# output = ['\t'.join(list(map(str,line))) for line in output]
	output_names = ['pp' + line[0].split('_pp')[-1] for line in output]
	filename = 'names-sorted-'+sort_type+'_'+filetype+'_'+exp
	open(path.data + filename,'w').close()
	[write(line,filename) for line in output_names]
	return output

	


	


		


	
