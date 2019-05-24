import utils
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

def make_n400_average(do_baseline_correction=True):
	'''loads dict with summed signal (selection of 14 channels) over all content words -300 to 1000 ms (word onset = 0)
	counter holds the number of word epochs
	the dict contains sums for the different experiments and suprisal values of the word epochs
	returns a dicts with the average over the 14 signal and an average over the content words (by dividing the sum by the counter)
	'''
	total,counter = utils.load_n400_dict()
	average = {}
	for k in counter.keys():
		average[k] = np.mean(total[k],0) / counter[k]
	if do_baseline_correction:average = baseline_correction(average)
	return average

def baseline_correction(average):
	output = {}
	for k in average.keys():
		output[k] = average[k] - np.mean(average[k][150:300])
	return output

def plot(name,figure = None,color ='black',linestyle = '-',ylim = (1,-3.1),xlim = (-300,1000),a = None, plot_axis = False, show_numbers = False):
	if a == None: a = make_n400_average()
	if type(figure) == matplotlib.figure.Figure: plt.figure(figure.number)
	elif figure == False: pass
	else: figure = plt.figure()
	plt.axis('off')
	x = np.arange(-300,1000)
	plt.ylim(ylim)
	plt.xlim(xlim)
	plt.plot(x,a[name],color = color,linestyle = linestyle)
	if plot_axis:
		plt.axhline(linewidth=1,color='black')
		plt.axvline(linewidth=1,linestyle='-',color='black')
		plt.axvline(300,color='tomato',linewidth=1,linestyle='--')
		plt.axvline(500,color='tomato',linewidth=1,linestyle='--')
	if show_numbers:
		plt.annotate('-300',xy= (-300,-0.1))
		plt.annotate('1000',xy= (800,-0.1))
		plt.annotate('-3',xy= (-180,-3))
		# plt.annotate('-1',xy= (-180,-1))
		plt.annotate(' 1',xy= (-180,1))

def get_names(a,experiment_name,register):
	if register == 'lp_':
		exp = [k for k in a.keys() if experiment_name in k and register in k and 'register' not in k and 'other' not in k]
	return exp

def make_names(experiment_name,register):
	names = []
	temp = 'low,middle,high'.split(',')
	for n in temp:
		names.append(register + n + experiment_name)
	return names

def plot_type(experiment_name = '-alls',register = 'lp_', plot_name = '',figure = None, ylim = None, show_numbers = True):
	a = make_n400_average()
	color = 'blue,orange,red'.split(',')
	# linestyles = [':','-.','-']
	names = make_names(experiment_name,register)
	for i, name in enumerate(names):
		if figure == None:
			figure = False if i > 0 else None
		plot_axis = True if i == 2 else False
		# plot(names[i],figure,color[i],a = a, plot_axis = plot_axis)
		if ylim != None:
			plot(names[i],figure,color[i],a = a, plot_axis = plot_axis, ylim = ylim, show_numbers = show_numbers)
			# plot(names[i],figure,linestyle= linestyles[i],a = a, plot_axis = plot_axis, ylim = ylim)
		# else: plot(names[i],figure,linestyle= linestyles[i],a = a, plot_axis = plot_axis)
		else: plot(names[i],figure,color[i],a = a, plot_axis = plot_axis, show_numbers = show_numbers)
		plt.title(plot_name)

def plot_alls():
	registers = 'lp_other1_,lp_,lp_register_'.split(',')
	rows,cols = 1, 3
	plt.figure()
	for i,r in enumerate(registers):
		plt.subplot(rows,cols,i+1)
		show_numbers = True if i == 0 else False
		plot_type(register= r,figure= False, show_numbers= show_numbers)
		if i == 0: plt.legend(('low','mid','high'),loc=1,bbox_to_anchor=(1.3,0.85))
	# plt.tight_layout()
	# plt.legend(('low','mid','high'))

def plot_exps():
	exps= '-ifadv,-o,-k'.split(',')
	rows,cols = 3, 1
	plt.figure()
	for i,e in enumerate(exps):
		plt.subplot(rows,cols,i+1)
		plot_type(experiment_name= e,register = 'lp_register_',figure= False,ylim = (1,-9))
	# plt.tight_layout()
	plt.legend(('low','mid','high'))
		
		



