#!/bin/bash
#
# Runs fit_pdfs.py on each pixel in a tarball containing galstar outputs
# for many lines of sight.
#
# Requires the following environmental variables to be set:
#    TMPDIR     = temporary directory
#    SCRIPTSDIR = directory with scripts
#    TARIN      = filename of tarball with galstar outputs
#    OUTFN      = file to output DM,Ar relation to for each pixel
#    ARGS       = optional arguments to pass to fit_pdfs.py
#

# Determine the temporary directory and working directory
workingdir=`pwd`
tmpdir=`readlink -m ${TMPDIR%/}`
scriptsdir=`readlink -m ${SCRIPTSDIR%/}`

# Determine absolute filename for output
outfn=`readlink -m $OUTFN`

# Remove output file if it already exists
#rm $outfn

# Get list of bin files from tarball
tarin=`readlink -m $TARIN`
binfilelist=`tar -tf $TARIN | grep .dat`

# Determine number of pixels in tarball
npix=0
for binfile in $binfilelist; do
	npix=`expr $npix + 1`
done

echo "Moving to $tmpdir"
cd $tmpdir

# Run fit_pdfs.py on each pixel
counter=1
for binfile in $binfilelist; do
	# Determine filenames corresponding to current pixel
	pixindex=${binfile%_DM_Ar.dat}
	statsfn="$pixindex.stats"
	binfn="${pixindex}_DM_Ar.dat"
	
	# Extract current pixel
	tar -xf $tarin $statsfn $binfn
	echo "$counter of $npix: Fitting l.o.s. reddening law for healpix pixel $pixindex ..."
	
	# Run fit_pdfs.py on this pixel
	$scriptsdir/fit_pdfs.py $binfn $statsfn -o $outfn $pixindex $ARGS
	
	# Remove temporary files
	rm $statsfn $binfn
	
	counter=`expr $counter + 1`
	echo ""
done

gzip -9 $outfn

cd $workingdir
echo "Done."
