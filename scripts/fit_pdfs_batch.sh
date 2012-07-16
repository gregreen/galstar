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
#    DIV        = # of processes working on this output file
#    PART       = part # of file this process is to complete
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

# If tarball is compresse and is not being shared (i.e. DIV = 1), unzip it
decompressed=0
if [ "${tarin##*.}" == "gz" ]; then
	#if [ $DIV -eq 1 ]; then
	echo "Decompressing $tarin ..."
	gzip -d $tarin
	tarin="${tarin%.gz}"
	echo "Done decompressing. Filename is now $tarin."
	decompressed=1
	#fi
fi

# Determine number of pixels in tarball
npix=0
for binfile in $binfilelist; do
	npix=`expr $npix + 1`
done

echo "Moving to $tmpdir"
cd $tmpdir

# Run fit_pdfs.py on each pixel
counter=1
thispart=1
partindex=1
partsize=`expr $npix / $DIV`
remainder=`expr $npix % $DIV`
remainderused=0

for binfile in $binfilelist; do
	# Determine whether to process this pixel
	if [ "$thispart" -gt "$partsize" ]; then
		if [ "$remainder" -gt "0" ]; then
			if [ "$remainderused" -eq "0" ]; then
				remainderused=1
				remainder=`expr $remainder - 1`
			else
				partindex=`expr $partindex + 1`
				thispart=2
				remainderused=0
			fi
		else
			partindex=`expr $partindex + 1`
			thispart=1
		fi
	else
		thispart=`expr $thispart + 1`
	fi
	
	if [ $partindex -eq $PART ]; then
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
		
		echo ""
	fi
	
	counter=`expr $counter + 1`
done

# Recompress galstar tarball, if it was earlier decompressed
if [ $decompressed -eq 1 ]; then
	echo "Recompressing $tarin."
	gzip -1 $tarin
fi

#gzip -9 $outfn

cd $workingdir
echo "Galstar has docked with the orbiting HQ."
