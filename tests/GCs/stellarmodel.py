#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#  
#  Copyright 2012 Greg Green <greg@greg-UX31A>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
import scipy.interpolate
import os

def bilinear(px, py, no_data=np.nan):
	'''Bilinear interpolated point at (px, py) on band_array
	example: bilinear(2790501.920, 6338905.159)'''
	ny, nx = band_array.shape
	# Half raster cell widths
	hx = gt[1]/2.0
	hy = gt[5]/2.0
	# Calculate raster lower bound indices from point
	fx = (px - (gt[0] + hx))/gt[1]
	fy = (py - (gt[3] + hy))/gt[5]
	ix1 = np.floor(fx).astype(np.int32)
	iy1 = np.floor(fy).astype(np.int32)
	# Special case where point is on upper bounds
	if fx == float(nx - 1):
		ix1 -= 1
	if fy == float(ny - 1):
		iy1 -= 1
	# Upper bound indices on raster
	ix2 = ix1 + 1
	iy2 = iy1 + 1
	# Test array bounds to ensure point is within raster midpoints
	if (ix1 < 0) or (iy1 < 0) or (ix2 > nx - 1) or (iy2 > ny - 1):
		return no_data
	# Calculate differences from point to bounding raster midpoints
	dx1 = px - (gt[0] + ix1*gt[1] + hx)
	dy1 = py - (gt[3] + iy1*gt[5] + hy)
	dx2 = (gt[0] + ix2*gt[1] + hx) - px
	dy2 = (gt[3] + iy2*gt[5] + hy) - py
	# Use the differences to weigh the four raster values
	div = gt[1]*gt[5]
	return (band_array[iy1,ix1]*dx2*dy2/div +
	        band_array[iy1,ix2]*dx1*dy2/div +
	        band_array[iy2,ix1]*dx2*dy1/div +
	        band_array[iy2,ix2]*dx1*dy1/div).astype(px.dtype)

class StellarModel:
	'''
	Loads the given stellar model, and provides access to interpolated
	colors on (Mr, FeH) grid.
	'''
	
	def __init__(self, template_fname):
		self.load_templates(template_fname)
	
	def load_templates(self, template_fname):
		'''
		Load in stellar template colors from an ASCII file. The colors
		should be stored in the following format:
		
		#
		# Arbitrary comments
		#
		# Mr    FeH   gr     ri     iz     zy
		# 
		-1.00 -2.50 0.5132 0.2444 0.1875 0.0298
		-0.99 -2.50 0.5128 0.2442 0.1873 0.0297
		...
		
		or something similar. A key point is that there be a row
		in the comments that lists the names of the colors. The code
		identifies this row by the presence of both 'Mr' and 'FeH' in
		the row, as above. The file must be whitespace-delimited, and
		any whitespace will do (note that the whitespace is not required
		to be regular).
		'''
		
		f = open(os.path.abspath(template_fname), 'r')
		row = []
		self.color_name = ['gr', 'ri', 'iz', 'zy']
		for l in f:
			line = l.rstrip().lstrip()
			if len(line) == 0:	# Empty line
				continue
			if line[0] == '#':	# Comment
				if ('Mr' in line) and ('FeH' in line):
					try:
						self.color_name = line.split()[3:]
					except:
						pass
				continue
			data = line.split()
			if len(data) < 6:
				print 'Error reading in stellar templates.'
				print 'The following line does not have the correct number of entries (6 expected):'
				print line
				return 0
			row.append([float(s) for s in data])
		f.close()
		template = np.array(row, dtype=np.float64)
		
		# Organize data into record array
		dtype = [('Mr','f4'), ('FeH','f4')]
		for c in self.color_name:
			dtype.append((c, 'f4'))
		self.data = np.empty(len(template), dtype=dtype)
		
		self.data['Mr'] = template[:,0]
		self.data['FeH'] = template[:,1]
		for i,c in enumerate(self.color_name):
			self.data[c] = template[:,i+2]
		
		self.MrFeH_bounds = [[np.min(self.data['Mr']), np.max(self.data['Mr'])],
		                     [np.min(self.data['FeH']), np.max(self.data['FeH'])]]
		
		# Produce interpolating class with data
		self.Mr_coords = np.unique(self.data['Mr'])
		self.FeH_coords = np.unique(self.data['FeH'])
		
		self.interp = {}
		for c in self.color_name:
			tmp = self.data[c][:]
			tmp.shape = (len(self.FeH_coords), len(self.Mr_coords))
			self.interp[c] = scipy.interpolate.RectBivariateSpline(
			                                      self.Mr_coords,
			                                      self.FeH_coords,
			                                      tmp.T,
			                                      kx=3,
			                                      ky=3,
			                                      s=0)

	def color(self, Mr, FeH, name=None):
		'''
		Return the colors, evaluated at the given points in
		(Mr, FeH)-space.
		
		Inputs:
		    Mr    float or array of floats
		    FeH   float or array of floats
		    name  string, or list of strings, with names of colors to
		          return. By default, all colors are returned.
		
		Output:
		    color  numpy record array of colors
		'''
		
		if name == None:
			name = self.get_color_names()
		elif type(name) == str:
			name = [name]
		
		if type(Mr) == float:
			Mr = np.array([Mr])
		elif type(Mr) == list:
			Mr = np.array(Mr)
		if type(FeH) == float:
			FeH = np.array([FeH])
		elif type(FeH) == list:
			FeH = np.array(FeH)
		
		dtype = []
		for c in name:
			if c not in self.color_name:
				raise ValueError('No such color in model: %s' % c)
			dtype.append((c, 'f4'))
		ret_color = np.empty(Mr.size, dtype=dtype)
		
		for c in name:
			ret_color[c] = self.interp[c].ev(Mr, FeH)
		
		return ret_color
	
	def get_color_names(self):
		'''
		Return the names of the colors in the templates.
		
		Ex.: For PS1 colors, this would return
		     ['gr', 'ri', 'iz', 'zy']
		'''
		
		return self.color_name



def main():
	model = StellarModel('../data/PScolors.dat')
	print model.get_color_names()
	
	print model.color(-0.5, -1.01, 'gr')
	
	
	return 0

if __name__ == '__main__':
	main()

