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
import matplotlib as mplib
import numpy as np
import healpy as hp


def lb2thetaphi(l, b):
	'''
	Convert Galactic (l, b) to spherical coordinates (theta, phi).
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


def thetaphi2lb(theta, phi):
	'''
	Convert spherical (theta, phi) to Galactic coordinates (l, b).
	Uses the physics convention, where theta is in the range [0, 180],
	and phi is in the range [0, 360].
	
	Input:
	    theta  Angle from the z-axis, in radians
	    phi    Angle from the prime meridian, in radians
	
	Output:
	    l      Galactic longitude, in degrees
	    b      Galactic latitude, in degrees
	'''
	return 180./np.pi * phi, 90. - 180./np.pi * theta


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


def healmap_rasterize(m, nside, nest=True, lb_bounds=[0., 360., -90., 90.], size=[1000,1000], center_gal=False, return_theta_phi=False):
	'''
	Rasterize a healpix map in Cartesian projection.
	
	Input:
	    m           Healpix map.
	    nside       Healpix nside parameter.
	    nest        True if map is stored in nested healpix ordering.
	    lb_bounds   (l_min, l_max, b_min, b_max). Default: all l,b included.
	    size        (x_size, y_size). # of pixels in each dimension.
	    center_gal  If True, place rasterized pixel closest to l=0 at
	                center of image.
	    
	    return_theta_phi  If True, spherical theta and phi (physics
	                      convention) for each rasterized pixel are
	                      returned.
	
	Output:
	    img         Rasterized image of healpix map in Cartesian projection.
	    
	If return_theta_phi is True, the following are also returned:
	    theta       Spherical theta coordinate in rad (physics convention).
	    phi         Spherical phi coordinate in rad (physics convention).
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
	
	# Center map on l=0
	if center_gal:
		phi[phi >= 360.] -= 360.
		shift = int(np.round(ysize/2. - np.unravel_index(np.argmin(np.abs(phi)), img.shape)[0]))
		np.roll(img, shift, axis=0)
	
	if return_theta_phi:
		return img, theta, phi
	else:
		return img


def healmap_to_axes(ax, m, nside, nest=True, lb_bounds=[0., 360., -90., 90.], size=[1000,1000], center_gal=False, **kwargs):
	'''
	Plot healpix map to the given pyplot axes. This function borrows
	liberally from Eddie Schlafly's util_efs.imshow.
	
	Input:
	    ax          Axes to which to plot healpix map.
	    m           Healpix map.
	    nside       Healpix nside parameter.
	    nest        True if map is stored in nested healpix ordering.
	    lb_bounds   (l_min, l_max, b_min, b_max). Default: all l,b included.
	    size        (x_size, y_size). # of pixels in each dimension.
	    center_gal  If True, place rasterized pixel closest to l=0 at
	                center of image.
	    **kwargs    Additional parameters to pass to pyplot.imshow.
	
	Output:
	    image       Image object returned by ax.imshow. This can be used,
	                for example, to create a colorbar.
	'''
	
	img, theta, phi = healmap_rasterize(m, nside, nest, lb_bounds, size, center_gal, True)
	
	lb_bounds_internal = np.array(lb_bounds)
	
	# Center map on l=0
	if center_gal:
		lb_bounds_internal[0] -= 180.
		lb_bounds_internal[1] -= 180.
	
	# Handle special case where the axes use the Mollweide projection
	if ax.name == 'mollweide':
		if not center_gal:
			lb_bounds_internal[0] -= 180.
			lb_bounds_internal[1] -= 180.
		if (np.abs(lb_bounds_internal[0] + 180.) > 0.001) or (np.abs(lb_bounds_internal[1] - 180.) > 0.001) or (np.abs(lb_bounds_internal[2] + 90.) > 0.001) or (np.abs(lb_bounds_internal[3] - 90.) > 0.001):
			print 'Warning: Mollweide projection requires lb_bounds = (0., 360., -90., 90.).'
		lb_bounds_internal = list(deg2rad(np.array(lb_bounds_internal)))
	
	# Plot to given axes
	if (ax.name != 'mollweide') and ('interpolation' not in kwargs):
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
	kwargs['extent'] = lb_bounds_internal
	image = ax.imshow(img.T, **kwargs)
	
	return image


def main():
	nside = 4
	nest = True
	npix = hp.pixelfunc.nside2npix(nside)
	#m = np.arange(npix)
	maps = np.random.random((6, npix))
	
	#hp.mollview(map=m)
	#plt.show()
	
	#plt.clf()
	
	mplib.rc('text', usetex=True)
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(7,5))
	image = None
	for i, m in enumerate(maps):
		ax = fig.add_subplot(2,3,i+1)#, projection='mollweide')
		y, x = np.unravel_index(i, (2,3))
		if y != 1:
			ax.set_xticklabels([])
		if x != 0:
			ax.set_yticklabels([])
		image = healmap_to_axes(ax, m, nside, nest=nest, size=(500,300), center_gal=True, clip_on=False, lb_bounds=[0., 360., -90., 90.])
	fig.subplots_adjust(wspace=0., hspace=0., right=0.88)
	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	cb = fig.colorbar(image, cax=cax)
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

