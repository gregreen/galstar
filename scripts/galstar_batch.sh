#!/bin/bash
#
# Runs galstar on all input files in the working directory,
# archiving the results in a tarball.
#
# Requires the following environmental variables to be set:
#    INFILE     = binary file containing input lines of sight
#    TMPDIR     = temporary directory
#    GALSTARDIR = directory containing galstar executable
#    SCRIPTSDIR = directory containing galstar scripts
#    TAROUT     = filename for tarball output
#

# Create an empty tar archive to store the output
tarfn=`readlink -m ${TAROUT%/}`
tar --create --file=$tarfn --files-from=/dev/null
echo "Writing binned pdfs and statistics to $tarfn."

# Determine working directory
workingdir=`pwd`
tmpdir=`readlink -m ${TMPDIR%/}`
galstardir=`readlink -m ${GALSTARDIR%/}`
scriptsdir=`readlink -m ${SCRIPTSDIR%/}`

# Determine filename for std. output/error
outfn="out-err.txt"

# Get the absolute path of the input file
infile=`readlink -m $INFILE`

# Determine number of pixels in the input
npix=`$scriptsdir/input_info.py $infile --npix`
maxpix=`expr $npix - 1`

echo "Moving to temporary folder $tmpdir"
cd $tmpdir

# Give each pixel in input file to galstar
for ((n=0; n<=$maxpix; n++)); do
	# Determine healpix number of this pixel
	pixindex=`$scriptsdir/input_info.py $infile --pix_index $n`
	
	# Name galstar output files by healpix number
	pixname=${infile%.in}
	statsfn="$pixindex.stats"
	binfn="${pixindex}_DM_Ar.dat"
	
	echo $binfn
	
	# Run galstar with the current pixel
	m=`expr $n + 1`
	echo "$m of $npix: Running galstar on healpix pixel $pixindex..."
	$galstardir/galstar $binfn:DM[5,20,120],Ar[0,25,1000] --statsfile $statsfn --infile $infile $n --errfloor 20 &> $outfn
	
	# Archive output, removing temporary files
	tar -rf $tarfn $binfn $statsfn
	rm $binfn $statsfn
done

# Add ASCII file containing std. out/err to tar archive and compress archive
tar -rf $tarfn $outfn
rm $outfn
gzip -9 $tarfn

cd $workingdir
echo "Done."
