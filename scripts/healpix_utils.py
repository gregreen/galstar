#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       healpix_utils.py
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

# TODO: Change to nest=True if query_lsd.py is run again.

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


def lb2thetaphi(l, b):
	'''
	Convert Galactic (l, b) to spherical coordinates, (theta, phi).
	Uses the physics convention, where theta is in the range [0, 180],
	and phi is in the range [0, 360].
	
	Input:
	    l      Galactic longitude, in degrees
	    b      Galactic latitude, in degrees
	Output:
	    theta  Angle from the z-axis, in radians
	    phi    Angle from the prime meridian, in radians
	'''
	return np.pi/180. * (90. - b), np.pi/180. * l


def deg2rad(theta):
	'''
	Convert from degrees to radians.
	
	Input:
	    theta    angle or numpy array of angles, in degrees
	Output:
	    angle(s) in radians
	'''
	return np.pi/180. * theta


def rad2deg(theta):
	'''
	Convert from radians to degrees.
	
	Input:
	    theta    angle or numpy array of angles, in radians
	Output:
	    angle(s) in degrees
	'''
	return 180./np.pi * theta


def healmap_to_axes(ax, m, nside, nest=True, lb_bounds=[0., 360., -90., 90.], size=[1000,1000], center_gal=False, **kwargs):
	'''
	Plot healpix map to the given pyplot axes. This function borrows
	liberally from Eddie Schlafly's util_efs.imshow.
	
	Input:
	    ax         Axes to which to plot healpix map
	    m          Healpix map
	    nside      Healpix nside parameter
	    nest       True if map is stored in nested healpix ordering.
	    lb_bounds  (l_min, l_max, b_min, b_max). Default: all l,b included.
	    size       (x_size, y_size). # of pixels in each dimension.
	    **kwargs   Additional parameters to pass to pyplot.imshow
	'''
	
	# Make grid of pixels to plot
	xsize, ysize = size
	l, b = np.mgrid[0:xsize, 0:ysize].astype(np.float32) + 0.5
	l = lb_bounds[0] + (lb_bounds[1] - lb_bounds[0]) * l / float(xsize)
	b = lb_bounds[2] + (lb_bounds[3] - lb_bounds[2]) * b / float(ysize)
	theta, phi = lb2thetaphi(l, b)
	del l, b
	
	# Convert coordinates to healpix pixels and create 2D map
	pix = hp.ang2pix(nside, theta, phi, nest=nest)
	img = m[pix]
	del pix
	img.shape = (xsize, ysize)
	if center_gal:
		phi[phi >= 360.] -= 360.
		shift = int(np.round(ysize/2. - np.unravel_index(np.argmin(np.abs(phi)), img.shape)[0]))
		np.roll(img, shift, axis=0)
		lb_bounds[0] -= 180.
		lb_bounds[1] -= 180.
	
	# Plot to axes provided
	if 'interpolation' not in kwargs:
		kwargs['interpolation'] = 'nearest'
	if 'vmin' not in kwargs:
		kwargs['vmin'] = np.min(img[np.isfinite(img)])
	if 'vmax' not in kwargs:
		kwargs['vmax'] = np.max(img[np.isfinite(img)])
	if 'aspect' not in kwargs:
		kwargs['aspect'] = 'auto'
	if 'origin' in kwargs:
		print "Ignoring option 'origin'."
	if 'extent' in kwargs:
		print "Ignoring option 'extent'."
	kwargs['origin'] = 'lower'
	kwargs['extent'] = lb_bounds
	ax.imshow(img.T, **kwargs)


def main():
	nside = 4
	npix = hp.pixelfunc.nside2npix(nside)
	m = np.arange(npix)
	
	hp.mollview(map=m)
	#plt.show()
	
	#plt.clf()
	
	fig = plt.figure(figsize=(7,5))
	ax = fig.add_subplot(2,1,1)
	healmap_to_axes(ax, m, nside, nest=False, size=(5000,100), center_gal=True)#, lb_bounds=[0., 360., -30., 30.])
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

