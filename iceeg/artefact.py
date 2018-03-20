import numpy as np

class artefact:
	def __init__(self,b):
		self.b = b
		self.name = windower.make_name(b)

	def __str__(self):
		m = '\nartefact\n'
		m += 'name\t\t\t'+str(self.name) + '\n'
		m += 'nblinks\t\t\t'+str(self.nblinks) + '\n'
		return m




	def load_blinks(self, offset = 1500):
		try:
			self.blinks_text= open(path.blinks + name + '_blink-model.classification').read()
			self.blink_peak_sample = np.array([int(line.split('\t')[2]) for line in self.blinks_text.split('\n')])
			self.nblinks = len(self.blink_peak_sample)
			self.blink_start = self.blink_peak_sample - offset
			self.blink_end = self.blink_peak_sample + offset
		except:
			self.blinks_text,self.blink_peak_sample,self.nblinks = 'NA','NA','NA'
			 self.blink_start, self.blink_end = 'NA','NA'



