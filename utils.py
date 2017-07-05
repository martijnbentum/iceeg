import numpy as np

'''general functions, not specific to an object

There are likely many functions now defined on objects that should be here
Work In Progress
'''


def make_attributes_available(obj, attr_name, attr_values,add_number = True,name_id = '',verbose = False):
	'''make attribute available on object as a property
	For example if attr_name is 'b' attr_value(s) can be accessed as: .b1 .b2 .b3 etc.

	Keywords:
	obj = the object the attributes should be added to
	attr_name = is the name the attributes should accessed by (see above)
	attr_values = list of values (e.g. a list of block objects)
	'''
	if type(attr_values) != list:
		# values should be provided in a list
		print('should be a list of value(s), received:',type(attr_values))
		return 0
	if len(attr_values) == 0:
		# Check for values
		print('should be a list with at leat 1 item, received empty list',attr_values)
		return 0

	# Make property name
	if add_number:
		# Add a number to the property name: .b1,.b2 etc.
		if verbose:
			print('Will add a number to:',attr_name,' for each value 1 ... n values')
		if name_id != '':
			print('Number is added to property, name id:',name_id,' will be ignored')
		if len(attr_values) > 1:
			property_names = [attr_name +str(i) for i in range(1,len(attr_values)+ 1)]
		else: property_names = [attr_name + '1']

	elif len(attr_values) > 1:
		print('add_number is False: you should only add one value otherwise you will overwrite values')
		return 0

	else:
		# Add name_id to property name
		if hasattr(obj,attr_name + name_id):
			print('object already had attribute:',attr_name,' will overwrite it with new value')
			print('Beware that discrepancies between property:', attr_name, ' and list of objects could arise')
			print('e.g. .pp1 could possibly not correspond to .pp[0]')
		property_names = [attr_name+name_id]

	# add value(s) to object 
	[setattr(obj,name,attr_values[i]) for i,name in enumerate(property_names)]

	#Add list of attribute names to object
	pn = 'property_names'
	if not attr_name.endswith('_'): pn = '_' + pn

	if hasattr(obj,attr_name + pn) and not add_number:
		# if no number the list of attribute names could already excist
		getattr(obj,attr_name + pn).extend(property_names)
	else:
		# otherwise create the list
		setattr(obj,attr_name + pn,property_names)

	if verbose:
		print('set the following attribute names:')
		print(' - '.join(property_names))


def make_events(start_end_sample_number_list):
	'''Make np array compatible with MNE EEG toolkit.

	assumes a list of lists with column of samplenumbers and a column of ids  int

	structure:   samplenumber 0 id_number
	dimension 3 X n_events.
	'''
	if set([len(line) for line in start_end_sample_number_list]):
		return np.asarray(output)	
