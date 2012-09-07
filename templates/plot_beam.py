#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#  plot_beam.py
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

import pyfits
import matplotlib.pyplot as plt
import matplotlib as mplib
import numpy as np
from plot_sdss_photometry import density_scatter


def plot_colors(d, model):
	# Correct for reddening
	A_coeff = np.array([3.634, 2.241, 1.568, 1.258, 1.074])
	m_g = d['g'] - A_coeff[0] * d['EBV']
	m_r = d['r'] - A_coeff[1] * d['EBV']
	m_i = d['i'] - A_coeff[2] * d['EBV']
	m_z = d['z'] - A_coeff[3] * d['EBV']
	m_y = d['y'] - A_coeff[4] * d['EBV']
	
	sel = (np.abs((m_g - m_r) - 0.315) < 10.02) & (np.abs((m_r - m_i) - 0.100) < 10.02)
	
	fig = plt.figure(figsize=(12,8), dpi=150)
	
	err = 0.02
	
	# Shifts
	diz = +0.045
	dzy = -0.020
	
	# Metallicity masks
	msel = np.abs(model['Mr'] - 3.76) < 100.01
	rich = msel & (model['FeH'] == 0.)
	poor = msel & (model['FeH'] == -2.5)
	
	# g-r vs r-i
	ax = fig.add_subplot(2,3,1)
	idx = sel & (m_g > 0.) & (m_r > 0.) & (m_i > 0.)
	idx = idx & (d['g_err'] < err) & (d['r_err'] < err) & (d['i_err'] < err)
	gr = m_g[idx] - m_r[idx]
	ri = m_r[idx] - m_i[idx]
	idx = (gr > - 0.5) & (gr < 2.) & (ri > -0.5) & (ri < 2.)
	density_scatter(ax, gr[idx], ri[idx], nbins=(80,80), threshold=3, s=1.5)
	ax.plot(model['gr'][rich], model['ri'][rich], 'k.', ms=0.3)
	#ax.plot(model['gr'][poor], model['ri'][poor], 'g.', ms=0.2)
	ax.set_xlabel(r'$g - r$', fontsize=16)
	ax.set_ylabel(r'$r - i$', fontsize=16)
	ax.set_xlim(-0.5, 2.)
	ax.set_ylim(-0.5, 2.)
	
	# r-i vs i-z
	ax = fig.add_subplot(2,3,2)
	idx = sel & (m_r > 0.) & (m_i > 0.) & (m_z > 0.)
	idx = idx & (d['r_err'] < err) & (d['i_err'] < err) & (d['z_err'] < err)
	ri = m_r[idx] - m_i[idx]
	iz = m_i[idx] - m_z[idx]
	idx = (ri > -0.5) & (ri < 2.) & (iz > -0.25) & (iz < 1.)
	density_scatter(ax, ri[idx], iz[idx], nbins=(80,80), threshold=3, s=1.5)
	ax.plot(model['ri'][rich], model['iz'][rich]+diz, 'k.', ms=0.3)
	#ax.plot(model['ri'][poor], model['iz'][poor]+0.030, 'g.', ms=0.2)
	ax.set_xlabel(r'$r - i', fontsize=16)
	ax.set_ylabel(r'$i - z$', fontsize=16)
	ax.set_xlim(-0.5, 2.)
	ax.set_ylim(-0.25, 1.)
	
	# i-z vs z-y
	ax = fig.add_subplot(2,3,3)
	idx = sel & (m_i > 0.) & (m_z > 0.) & (m_y > 0.)
	idx = idx & (d['i_err'] < err) & (d['z_err'] < err) & (d['y_err'] < err)
	iz = m_i[idx] - m_z[idx]
	zy = m_z[idx] - m_y[idx]
	idx = (iz > -0.25) & (iz < 1.) & (zy > -0.2) & (zy < 0.6)
	density_scatter(ax, iz[idx], zy[idx], nbins=(80,80), threshold=3, s=1.5)
	ax.plot(model['iz'][rich]+diz, model['zy'][rich]+dzy, 'k.', ms=0.3)
	#ax.plot(model['iz'][poor]+0.030, model['zy'][poor]-0.035, 'g.', ms=0.2)
	ax.set_xlabel(r'$i - z$', fontsize=16)
	ax.set_ylabel(r'$z - y$', fontsize=16)
	ax.set_xlim(-0.25, 1.)
	ax.set_ylim(-0.2, 0.6)
	
	# g-r vs z-y
	ax = fig.add_subplot(2,3,4)
	idx = (m_g > 0.) & (m_r > 0.) & (m_z > 0.) & (m_y > 0.)
	idx = idx & (d['g_err'] < err) & (d['r_err'] < err) & (d['z_err'] < err) & (d['y_err'] < err)
	gr = m_g[idx] - m_r[idx]
	zy = m_z[idx] - m_y[idx]
	idx = (gr > - 0.5) & (gr < 2.) & (zy > -0.2) & (zy < 0.6)
	density_scatter(ax, gr[idx], zy[idx], nbins=(80,80), threshold=3, s=1.5)
	ax.plot(model['gr'][rich], model['zy'][rich]+dzy, 'k.', ms=0.3)
	ax.set_xlabel(r'$g - r$', fontsize=16)
	ax.set_ylabel(r'$z - y$', fontsize=16)
	ax.set_xlim(-0.5, 2.)
	ax.set_ylim(-0.2, 0.6)
	
	# g-r vs i-z
	ax = fig.add_subplot(2,3,5)
	idx = (m_g > 0.) & (m_r > 0.) & (m_i > 0.) & (m_z > 0.)
	idx = idx & (d['g_err'] < err) & (d['r_err'] < err) & (d['i_err'] < err) & (d['z_err'] < err)
	gr = m_g[idx] - m_r[idx]
	iz = m_i[idx] - m_z[idx]
	idx = (gr > - 0.5) & (gr < 2.) & (iz > -0.25) & (iz < 1.)
	density_scatter(ax, gr[idx], iz[idx], nbins=(80,80), threshold=3, s=1.5)
	ax.plot(model['gr'][rich], model['iz'][rich]+diz, 'k.', ms=0.3)
	ax.set_xlabel(r'$g - r$', fontsize=16)
	ax.set_ylabel(r'$i - z$', fontsize=16)
	ax.set_xlim(-0.5, 2.)
	ax.set_ylim(-0.25, 1.)
	
	# r-i vs z-y
	ax = fig.add_subplot(2,3,6)
	idx = (m_g > 0.) & (m_r > 0.) & (m_z > 0.) & (m_y > 0.)
	idx = idx & (d['r_err'] < err) & (d['i_err'] < err) & (d['z_err'] < err) & (d['y_err'] < err)
	ri = m_r[idx] - m_i[idx]
	zy = m_z[idx] - m_y[idx]
	idx = (ri > - 0.5) & (ri < 2.) & (zy > -0.2) & (zy < 0.6)
	density_scatter(ax, ri[idx], zy[idx], nbins=(80,80), threshold=3, s=1.5)
	ax.plot(model['ri'][rich], model['zy'][rich]+dzy, 'k.', ms=0.3)
	ax.set_xlabel(r'$r - i$', fontsize=16)
	ax.set_ylabel(r'$z - y$', fontsize=16)
	ax.set_xlim(-0.5, 2.)
	ax.set_ylim(-0.2, 0.6)
	
	fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.25)


def load_templates(fname, max_r=12):
	f = open(fname, 'r')
	row = []
	for l in f:
		line = l.rstrip().lstrip()
		if len(line) == 0:	# Empty line
			continue
		if line[0] == '#':	# Comment
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
	template = template[template[:,0] <= max_r]
	
	tmp = np.empty(len(template), dtype=[('Mr','f4'), ('FeH','f4'),
	                                     ('gr','f4'), ('ri','f4'),
	                                     ('iz','f4'), ('zy','f4')])
	tmp['Mr'] = template[:,0]
	tmp['FeH'] = template[:,1]
	tmp['gr'] = template[:,2]
	tmp['ri'] = template[:,3]
	tmp['iz'] = template[:,4]
	tmp['zy'] = template[:,5]
	
	return tmp


def main():
	data = pyfits.getdata('N2.fits')
	model = load_templates('../data/PScolors.dat.bak')
	
	mplib.rc('text', usetex=True)
	plot_colors(data, model)
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

