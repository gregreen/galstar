#!/usr/bin/python
from numpy import *;
import numpy;
import matplotlib.pyplot as plt;
from matplotlib import *;

def plotpdf(imgfn, fn, xname, yname):
	# convert SM-formatted image to an array
	x, y, p = numpy.lib.io.loadtxt(fn, usecols=(0, 1, 2), unpack=True)
	xs = x.copy(); xs.sort(); dx = xs[1:xs.size] - xs[0:xs.size-1]; dx = dx.max();
	ys = y.copy(); ys.sort(); dy = ys[1:ys.size] - ys[0:ys.size-1]; dy = dy.max();
	i = ((x - xs[0]) / dx).round().astype(int);
	j = ((y - ys[0]) / dy).round().astype(int);
	nx = i.max() + 1;
	ny = j.max() + 1;
	img = zeros([nx, ny]); img[:,:] = p.min();
	img[i,j] = p;

	# Plot the image
	plt.clf();
	plt.imshow(img.transpose(), origin='lower', aspect='auto', interpolation='bilinear', cmap='hot', extent=(x.min(),x.max(),y.min(),y.max()));

	# title and legend
	plt.title('$\ln P(' + xname + ',' + yname + ')$, normalized to peak')
	plt.colorbar();
	plt.xlabel('$' + xname + '$')
	plt.ylabel('$' + yname + '$')

	# Maximum likelihood point, and expectation value
	imax = p.argmax();
	plt.scatter(x[imax], y[imax], s=40, marker='x');

	xmean = (exp(p)*x).sum() / exp(p).sum();
	ymean = (exp(p)*y).sum() / exp(p).sum();
	plt.scatter(xmean, ymean, s=40, marker='o');

	# Contouring at levels of accumulated probability
	p2 = p.copy(); p2.sort(); cum = (exp(p2)/sum(exp(p2))).cumsum(); 
	cont90 = p2[abs(cum - 0.9).argmin()];
	cont50 = p2[abs(cum - 0.5).argmin()];
	cont10 = p2[abs(cum - 0.1).argmin()];
	cont01 = p2[abs(cum - 0.01).argmin()];
	levs = [cont01, cont10, cont50, cont90];
	clabels = {cont01: "1%", cont10: "10%", cont50: "50%", cont90: "90%"}
	cont = plt.contour(img.transpose(), levs, origin='lower', colors='black', extent=(x.min(),x.max(),y.min(),y.max()), hold=True);
	plt.clabel(cont, fmt=clabels, fontsize=8)

	# Save image
	plt.savefig(imgfn, transparent=True);

#print sys.argv;
if len(sys.argv) != 5:
	print "Usage: " + sys.argv[0] + " <prob.txt> <xname> <yname> <prob.png|prob.eps>";
	sys.exit(-1);

#plotpdf('prob_MrFeH.txt', 'M_r', '[Fe/H]')
plotpdf(sys.argv[4], sys.argv[1], sys.argv[2], sys.argv[3]);

