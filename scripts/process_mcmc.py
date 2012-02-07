#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       plot-mcmc.py
#       
#       Gregory Green, Jun. 2011 <greg@greg-G53JW>

from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, NullLocator
from matplotlib import rc

class TMCMC:
	def __init__(self, filename, ascii=True):
		self.load_from_file(filename, ascii)
	
	def load_from_file(self, filename, ascii=True):
		f = open(fname+'.mcmc', 'rb')
		mcmc = f.read()
		f.close()


'''
// Write some basic information about the chain first
		write_binary_data<unsigned int>(outfile, N_dim, offset);
		write_binary_data<unsigned int>(outfile, length, offset);
		write_binary_data<unsigned int>(outfile, N_items_tot, offset);
		write_binary_data<unsigned int>(outfile, max_length, offset);
		write_binary_data<unsigned int>(outfile, index_of_best, offset);
		
		// Write the statistics of the chain
		for(unsigned int i=0; i<mean.size(); i++) { write_binary_data<double>(outfile, mean.at(i), offset); }
		for(unsigned int i=0; i<covariance.size(); i++) {
			for(unsigned int k=0; k<covariance.at(i).size(); k++) { write_binary_data<double>(outfile, covariance.at(i).at(k), offset); }
		}
		for(unsigned int i=0; i<N_dim; i++) {
			double element = scale.get_element(i);
			write_binary_data<double>(outfile, element, offset);
		}
		
		// Write the full chain
		for(unsigned int i=0; i<chain.size(); i++) {
			for(unsigned int k=0; k<N_dim; k++) {
				double element = chain.at(i).get_element(k);
				write_binary_data<double>(outfile, element, offset);
			}
		}
		for(unsigned int i=0; i<count.size(); i++) { write_binary_data<unsigned int>(outfile, count.at(i), offset); }
		for(unsigned int i=0; i<pi.size(); i++) { write_binary_data<double>(outfile, pi.at(i), offset); }
'''

def main():
	
	return 0

if __name__ == '__main__':
	main()

