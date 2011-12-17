#!/usr/bin/python

import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib as mplib
import sys
import argparse
from operator import itemgetter
from math import isnan
import galstarutils


# Load true parameter values from ascii file
def load_true_values(fn):
	params = np.loadtxt(fn, usecols=(0,1,2,3))
	return params

# Load SM-formatted image and return an array
def loadimage(fn):
	x, y, p = np.loadtxt(fn, usecols=(0, 1, 2), unpack=True)
	for i,p_i in enumerate(p):
		if isnan(p_i): p[i] = -999.
	#print 'loading %s' % fn, p[0:10]
	# Sort x and y and get dx and dy
	xs = x.copy(); xs.sort(); dx = xs[1:xs.size] - xs[0:xs.size-1]; dx = dx.max();
	ys = y.copy(); ys.sort(); dy = ys[1:ys.size] - ys[0:ys.size-1]; dy = dy.max();
	# Nearest-neighbor interpolation
	i = ((x - xs[0]) / dx).round().astype(int)
	j = ((y - ys[0]) / dy).round().astype(int)
	# Determine width and height of image
	nx = i.max() + 1
	ny = j.max() + 1
	# Fill in the image
	img = np.zeros([nx, ny]); img[:,:] = p.min();
	img[i,j] = p
	return img, x, y, p

def plotimg(img, x, y, p, ax, xname=None, yname=None, xlim=(None,None), ylim=(None,None), params=None):
	bounds = [x.min(), x.max(), y.min(), y.max()]
	for i in range(2):
		if xlim[i] != None: bounds[i] = xlim[i]
		if ylim[i] != None: bounds[i+2]= ylim[i]
	ax.imshow(img.transpose(), origin='lower', aspect='auto', interpolation='bilinear', cmap='hot', extent=(x.min(),x.max(),y.min(),y.max()))
	# Maximum likelihood point, and expectation value
	imax = p.argmax()
	ax.scatter(x[imax], y[imax], s=40, marker='x')
	p_filtered = p.copy()
	p_min = p.min()
	for i in range(len(p_filtered)):
		if p_filtered[i] == p_min: p_filtered[i] = -999.
	xmean = (np.exp(p_filtered)*x).sum() / np.exp(p_filtered).sum()
	ymean = (np.exp(p_filtered)*y).sum() / np.exp(p_filtered).sum()
	#if not (isnan(xmean) or isnan(ymean)):
	ax.scatter(xmean, ymean, s=40, marker='o')
	if params != None:
		ax.scatter(params[0], params[1], s=40, marker='d', c='r')
	# Contouring at levels of accumulated probability
	p2 = p.copy(); p2.sort(); cum = (np.exp(p2)/sum(np.exp(p2))).cumsum();
	cont90 = p2[abs(cum - 0.9).argmin()]
	cont50 = p2[abs(cum - 0.5).argmin()]
	cont10 = p2[abs(cum - 0.1).argmin()]
	cont01 = p2[abs(cum - 0.01).argmin()]
	levs = [cont01, cont10, cont50, cont90]
	clabels = {cont01: "1%", cont10: "10%", cont50: "50%", cont90: "90%"}
	cont = ax.contour(img.transpose(), levs, origin='lower', colors='black', extent=(x.min(),x.max(),y.min(),y.max()), hold=True)
	ax.clabel(cont, fmt=clabels, fontsize=8)
	# Set canvas size
	ax.set_xlim(bounds[0:2])
	ax.set_ylim(bounds[2:])
	# Set axes labels
	if xname != None: ax.set_xlabel(r'$\mathrm{%s}$'%xname)
	if xname != None: ax.set_ylabel(r'$\mathrm{%s}$'%yname)

# Draw a figure with multiple plots, laid out in a manner determined by <shape>
def make_figure(fn_list, img_fname, shape, xname, yname, xlim=(None,None), ylim=(None,None), params_list=None):
	fig = plt.figure(figsize=(8.5,11.))
	ax = []
	for i,fname in enumerate(fn_list):
		img, x, y, p = loadimage(fname)
		ax.append(fig.add_subplot(shape[0], shape[1], i+1, axisbg='k'))
		if params_list != None:
			params_i = params_list[i]
		else:
			params_i = None
		plotimg(img, x, y, p, ax[-1], xname, yname, xlim, ylim, params_i)
	fig.suptitle(r'$\ln \mathrm{P}(' + xname + '\, , \,' + yname + ') \mathrm{, \ normalized \ to \ peak}$', y=0.95, fontsize=16)
	fig.savefig(img_fname, transparent=False, dpi=300)
	plt.close(fig)

def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='plotpdf', description='Plots posterior distributions produced by galstar', add_help=True)
	parser.add_argument('files', nargs='+', type=str, help='Input posterior distributions')
	parser.add_argument('--truth', type=str, default=None, help='File containing true parameter values')
	parser.add_argument('--converged', nargs='+', type=str, help='Filter out nonconverged stars using provided stats files')
	parser.add_argument('--output', type=str, required=True, help='Output image filename base (without extension)')
	parser.add_argument('--imgtype', type=str, default='png', choices=('png','pdf','eps'), help='Output image filetype')
	parser.add_argument('--shape', nargs=2, type=int, default=(1,1), help='# of rows and columns in figure')
	parser.add_argument('--xname', type=str, default='X', help='Name of x-axis')
	parser.add_argument('--yname', type=str, default='Y', help='Name of y-axis')
	parser.add_argument('--xmin', type=float, default=None, help='Lower bound of x in plots')
	parser.add_argument('--xmax', type=float, default=None, help='Upper bound of x in plots')
	parser.add_argument('--ymin', type=float, default=None, help='Lower bound of y in plots')
	parser.add_argument('--ymax', type=float, default=None, help='Upper bound of y in plots')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Sort filenames
	files = galstarutils.sort_filenames(values.files)
	
	# Load true stellar parameters
	params = None
	if values.truth != None:
		params = np.loadtxt(values.truth, usecols=(0,1,2,3))
	
	# Load stats files and filter out nonconverged stars
	stats_fn = None
	N = len(files)
	convergence_filter = np.empty(N, dtype=bool)
	convergence_filter.fill(True)
	if values.converged != None:
		stats_fn = galstarutils.sort_filenames(values.converged)
		for i,fn in enumerate(stats_fn):
			converged, mean, cov, ML_dim, ML = galstarutils.read_stats(fn)
			convergence_filter[i] = converged
	files_tmp = []
	for i,f in enumerate(files):
		if convergence_filter[i]:
			files_tmp.append(f)
	files = files_tmp
	if params != None:
		params = params[convergence_filter]
	
	# Make figures
	mplib.rc('text', usetex=True)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	
	# Determine # of figures to make
	N_per_fig = values.shape[0]*values.shape[1]
	N_figs = int(len(values.files)/N_per_fig)
	print N_per_fig, N_figs, len(values.files)
	if N_figs*N_per_fig < len(values.files):
		N_figs += 1
	
	# Get x and y bounds for plots
	xlim = (values.xmin, values.xmax)
	ylim = (values.ymin, values.ymax)
	
	for i in range(N_figs):
		# Determine range of plots to put on this figure
		i_min = i*N_per_fig
		i_max = i_min + N_per_fig
		if i_max > len(files): i_max = len(files)
		print 'Plotting files %d through %d...' % (i_min+1, i_max)
		fn_list = files[i_min:i_max]
		if params != None:
			params_list = params[i_min:i_max+1]
		else:
			params_list = None
		
		print fn_list
		
		# Determine output filename for this figure
		img_fname = str(values.output + '_' + str(i) + '.' + values.imgtype)
		
		# Generate this figure
		make_figure(fn_list, img_fname, values.shape, values.xname, values.yname, xlim, ylim, params_list)
	
	print 'Done.'
	return 0

if __name__ == '__main__':
	main()
