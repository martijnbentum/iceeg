from setuptools import setup

setup(name='ifadv_cgn_scripts',
	  version='0.1',
	  discription='Analysis script for EEG data of participants listening to speech from different corpora',
	  url='https://github.com/martijnbentum/ifadv_cgn_scripts',
	  author='Martijn Bentum',
	  author_email='bentummartijn@gmail.com',
	  license='MIT',
	  packages=['ifadv_cgn_scripts'],
	  install_requires=['numpy','mne']
	  zip_safe=False)

