#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       sdss2ps.py
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

# TODO:
#   1. Transform luminosity function in similar manner


import sys, argparse
from os.path import abspath

import numpy as np
from scipy.interpolate import griddata, interp1d

import ps


def load_SDSS_templates(fname):
	'''Load Mario's stellar color template library from file, returning the absolute magnitudes and metallicities.'''
	
	MrFeH = np.loadtxt(abspath(fname), usecols=(0,1), dtype=np.float32)
	colors = np.loadtxt(abspath(fname), usecols=(2,3,4,5), dtype=np.float32)	# ug gr ri iz
	
	N = MrFeH.shape[0]
	ugriz = np.empty(N, dtype=[('u','f4'), ('g','f4'), ('r','f4'), ('i','f4'), ('z','f4')])
	
	ugriz['r'] = MrFeH[:,0]
	ugriz['g'] = ugriz['r'] + colors[:,1]	# r = r + (g-r)
	ugriz['u'] = ugriz['g'] + colors[:,0]	# u = g + (u-g)
	ugriz['i'] = ugriz['r'] - colors[:,2]	# i = r - (r-i)
	ugriz['z'] = ugriz['i'] - colors[:,3]	# z = i - (i-z)
	
	# Return ugriz, MrFeH
	return ugriz, MrFeH


def interp_to_grid(data, old_grid, new_grid):
	'''Interpolate data on a 2D grid to a new 2D grid.'''
	
	grid = griddata(old_grid, data, new_grid, method='cubic')
	return grid


def save_PS_templates(fname, PSgrizy, MrFeH):
	'''Transform PanSTARRS magnitudes to colors and save as a template library.'''
	
	# Compute the PanSTARRS colors from the given absolute magnitudes
	N = MrFeH.shape[0]
	colors = np.empty((N,4), dtype=np.float32)
	for i in range(4):
		colors[:,i] = PSgrizy[:,i] - PSgrizy[:,i+1]	# gr ri iz zy
	
	# Clip interpolation grid to maximum PS r-band magnitude (since less than maximum SDSS r)
	#PS_rmin = np.min(PSgrizy[:,1])
	PS_rmax = np.max(PSgrizy[:,1])
	#print PS_rmin, PS_rmax
	mask = (MrFeH[:,0] < PS_rmax)
	
	# Interpolate back to the original grid in (Mr, FeH), albeit clipped as described above
	PS_grid = np.array([PSgrizy[:,1], MrFeH[:,1]]).T
	colors = interp_to_grid(colors, PS_grid, MrFeH[mask])
	
	# Organize new template output into one array
	N = np.sum(mask)
	output = np.empty((N,6), dtype=np.float32)
	output[:,0] = MrFeH[mask,0]
	output[:,1] = MrFeH[mask,1]
	for i in range(4):
		output[:,i+2] = colors[:,i]
	
	# Write to an ASCII file
	header = """# Transformed to the PS filter set from Zeljko and Juric's SDSS template
# library (MSandRGBcolors_v1.3.dat), using Eddie Schlafly's
# pssdsstransformall routine in ps.py. For details, see Greg Green's
# sdss2ps.py. For more information on the original SDSS template
# library, see the header in MSandRGBcolors_v1.3.dat.
# 
# Mr    FeH   gr     ri     iz     zy
# 
"""
	np.savetxt(fname, output, fmt='%.2f %.2f %.4f %.4f %.4f %.4f')
	f = open(fname, 'r')
	tmp = f.read()
	f.close()
	f = open(fname, 'w')
	f.write(header)
	f.write(tmp)
	f.close()


def save_PS_lf(fname, SDSSlf, SDSSMrFeH, PSr):
	'''Transform SDSS luminosity function to a PanSTARRS luminosity function and save result.'''
	
	# Choose row from MrFeH where FeH is closest to solar
	FeH_closest = np.min(np.abs(SDSSMrFeH[:,1]))
	FeH_mask = (SDSSMrFeH[:,1] == FeH_closest)
	SDSSr = SDSSMrFeH[FeH_mask][:,0]
	
	# Transform SDSS r magnitudes in luminosity function to PS r magnitudes
	PSr_from_SDSSr = interp1d(SDSSr, PSr[FeH_mask])
	PSr_lf = PSr_from_SDSSr(SDSSlf[:,0])
	PSlf_interp = interp1d(PSr_lf, SDSSlf[:,1])
	
	# Clip interpolation grid to maximum PS r-band magnitude (since less than maximum SDSS r)
	PSr_max = np.max(PSr_lf)
	mask = (SDSSlf[:,0] <= PSr_max)
	N = np.sum(mask)
	
	# Interpolate back to original set of r magnitudes in luminosity function, albeit clipped to max. PS r
	output = np.empty((N,2), dtype=np.float32)
	output[:,0] = SDSSr[mask]
	output[:,1] = PSlf_interp(SDSSr[mask])
	
	# Write to an ASCII file
	header = """# Transformed to the PS r-band from Zeljko and Juric's SDSS luminosity
# function (MrLF.MSandRGB_v1.0.dat), using Eddie Schlafly's
# pssdsstransformall routine in ps.py. For details, see Greg Green's
# sdss2ps.py. For more information on the original SDSS template
# library, see the header in MrLF.MSandRGB_v1.0.dat.
# 
#  Mr        LF
# (AB) (stars/pc^3/mag)
"""
	np.savetxt(fname, output, fmt='%.2f %.3g')
	f = open(fname, 'r')
	tmp = f.read()
	f.close()
	f = open(fname, 'w')
	f.write(header)
	f.write(tmp)
	f.close()


def main():
	parser = argparse.ArgumentParser(prog='sdss2ps.py', description='Transform template library from SDSS colors to PanSTARRS colors.', add_help=True)
	parser.add_argument('SDSStemp', type=str, help="Mario's SDSS template library.")
	parser.add_argument('PStemp', type=str, help='Output filename for PanSTARRS template library.')
	parser.add_argument('SDSSlf', type=str, help="Mario's SDSS luminosity function.")
	parser.add_argument('PSlf', type=str, help='Output filename for PanSTARRS luminosity function.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	SDSSugriz, MrFeH = load_SDSS_templates(values.SDSStemp)
	PSgrizy = ps.pssdsstransformall(SDSSugriz)
	save_PS_templates(values.PStemp, PSgrizy, MrFeH)
	
	SDSSlf = np.loadtxt(abspath(values.SDSSlf), usecols=(0,1), dtype=np.float32)
	save_PS_lf(values.PSlf, SDSSlf, MrFeH, PSgrizy[:,1])
	
	return 0

if __name__ == '__main__':
	main()

