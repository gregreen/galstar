#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       iterators.py
#       
#       Copyright 2012 Greg <greg@greg-G53JW>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#       

import numpy as np


########################################################################
#
# Grouping iterators
#
# The following iterators group data by keys, sometimes sorted by key.
#
########################################################################

class data_by_key(object):
	'''Returns blocks of data having the same key, sorted by key.'''
	
	def __init__(self, key, data):
		self.data = data
		
		self.indices = key.argsort()
		self.key = key[self.indices]
		self.newblock = np.concatenate((np.where(np.diff(self.key))[0] + 1, [data.size]))
		
		self.start_index = 0
		self.end_index = self.newblock[0]
		self.block_num = 0
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.block_num == self.newblock.size:
			raise StopIteration
		else:
			block_indices = self.indices[self.start_index:self.end_index]
			block_key = self.key[self.start_index]
			
			self.block_num += 1
			if self.block_num < self.newblock.size:
				self.start_index = self.end_index
				self.end_index = self.newblock[self.block_num]
			
			return block_key, self.data[block_indices]


class index_by_key(object):
	'''Returns sets of indices referring to each value of the key,
	in ascending order of key value.'''
	
	def __init__(self, key):
		self.indices = key.argsort()
		self.key = key[self.indices]
		self.newblock = np.concatenate((np.where(np.diff(self.key))[0] + 1, [key.size]))
		
		self.start_index = 0
		self.end_index = self.newblock[0]
		self.block_num = 0
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.block_num == self.newblock.size:
			raise StopIteration
		else:
			block_indices = self.indices[self.start_index:self.end_index]
			block_key = self.key[self.start_index]
			
			self.block_num += 1
			if self.block_num < self.newblock.size:
				self.start_index = self.end_index
				self.end_index = self.newblock[self.block_num]
			
			return block_key, block_indices


class index_by_unsortable_key(object):
	'''Returns sets of indices referring to each value of the key,
	where the key is not sortable (does not admit < or > operators).'''
	
	def __init__(self, key):
		self.key = key
		self.unused = np.ones(len(key), dtype=np.bool)
	
	def __iter__(self):
		return self
	
	def next(self):
		if np.all(~self.unused):
			raise StopIteration
		else:
			block_indices = []
			unused_indices = np.where(self.unused)[0]
			for i in unused_indices:
				if np.all(self.key[i] == self.key[unused_indices[0]]):
					block_indices.append(i)
					self.unused[i] = False
			return self.key[unused_indices[0]], block_indices


########################################################################
#
# String iterators
#
# The following iterators yield parts of strings in various ways.
#
########################################################################

class block_string_by_comments(object):
	'''
	Returns contiguous lines of string between commented lines.
	
	Inputs:
	    string              String to split into blocks
	    comments            Comment marker (Default '#')
	    ignore_whitespace   Ignore leading whitespace before comments
	    ignore_empty_lines  Ignore empty lines bewteen comments
	                        (Never return a set of empty lines as a block)
	
	Outputs:
	    One block of string at a time. Each block is bracketed by
	    commented lines.
	'''
	
	def __init__(self, string, comments='#', ignore_whitespace=True,
	             ignore_empty_lines=True):
		self.string = string
		self.comments = comments
		self.ignore_whitespace = ignore_whitespace
		self.ignore_empty_lines = ignore_empty_lines
		self.lines = string.splitlines(True)
		self.start_line = 0
		self.end_pos = 0
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.start_line >= len(self.lines):
			raise StopIteration
		
		self.start_pos = self.end_pos
		
		# Find first non-commented line
		for line in self.lines[self.start_line:]:
			self.end_pos += len(line)
			self.start_line += 1
			if self.ignore_whitespace:
				line = line.lstrip()
			if ((self.ignore_empty_lines and (len(line) == 0))
			                 or line.startswith(self.comments)):
				if self.start_line >= len(self.lines):
					raise StopIteration
			else:
				break
		
		#print '// First non-commented line:'
		#print self.lines[self.start_line-1]
		#print '//'
		
		# Find next commented line, or end of string
		for line in self.lines[self.start_line:]:
			line_length = len(line)
			if self.ignore_whitespace:
				line = line.lstrip()
			if line.startswith(self.comments):
				#print '// Next commented line:'
				#print line
				#print '//'
				break
			self.end_pos += line_length
			self.start_line += 1
		
		#print self.start_pos, self.end_pos
		return self.string[self.start_pos:self.end_pos]


def main():
	x = np.array([0, 1, 1, -3, 5, 5, 5, -3, 0, 0])
	y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	
	for key, block_indices in index_by_key(x):
		print key, y[block_indices]
	
	print ''
	
	x = [np.array([1,2,3]), np.array([2,3,4]), np.array([1,2,3]), np.array([4,6,-1]), np.array([4,6,-1]), np.array([4,6,-2]), np.array([1,2,3])]
	y = np.array([1, 2, 3, 4, 5, 6, 7])
	for key, block_indices in index_by_unsortable_key(x):
		print key, y[block_indices]
	
	print ''
	
	string = '''
# Commented line

some data
more data

# Another commented line
# Some more comments
more data
and even more
and yet more data
another row

a row after a space

another row after a space


a row after two spaces



a row after three spaces

#
# A comment
#

#
# A second comment
#

Data

# Final comment
'''
	
	for i,block in enumerate(block_string_by_comments(string)):
		print ''
		print '======================================================'
		print 'Block #%d' % i
		print block
		
	return 0

if __name__ == '__main__':
	main()

