#!/bin/bash

#
# Runs fit_pdfs.py on each pixel in a tarball containing galstar outputs
# for many lines of sight.
#
# Requires the following environmental variables to be set:
#    TMPDIR     = temporary directory
#    SCRIPTSDIR = directory with scripts
#    TARFN      = filename of tarball with galstar outputs
#    OUTFN      = file to output DM,Ar relation to for each pixel
#    ARGS       = optional arguments to pass to fit_pdfs.py
#

# Determine the temporary directory and working directory
workingdir=`pwd`
tmpdir=`readlink -m ${TMPDIR%/}`
scriptsdir=`readlink -m ${SCRIPTSDIR%/}`

# Determine absolute filename for output
outfn=`readlink -m $OUTFN`

# Get list of bin files from tarball
tarfn_abs=`readlink -m $TARFN`
binfilelist=`tar -tf $TARFN | grep .dat.gz`

# Determine number of pixels in tarball
npix=0
for binfile in $binfilelist; do
	npix=`expr $npix + 1`
done

cd $tmpdir

# Run fit_pdfs.py on each pixel
counter=1
for binfile in $binfilelist; do
	# Determine filenames corresponding to current pixel
	pixname=${binfile%_DM_Ar.dat.gz}
	statsfn="$pixname.stats"
        binfn="${pixname}_DM_Ar.dat.gz"
	
	# Extract current pixel
	tar -xf $tarfn_abs $statsfn $binfn
	echo "$counter of $npix: Fitting l.o.s. reddening law for $pixname ..."
	echo $statsfn
	echo $binfn
	
	# Run fit_pdfs.py on this pixel
	echo $pixname >> $outfn
	$scriptsdir/fit_pdfs.py $binfn $statsfn $ARGS >> $outfn
	echo "" >> $outfn
	
	# Remove temporary files
	rm $statsfn $binfn
	
	counter=`expr $counter + 1`
	echo ""
done

#gzip -9 $outfn

#cd $workingdir
