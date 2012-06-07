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
npix=`$scriptsdir/npix_in_input.py $infile`
maxpix=`expr $npix - 1`

echo "Moving to temporary folder $tmpdir"
cd $tmpdir

# Give each pixel in input file to galstar
for n in {0..maxpix}; do
	# Determine filenames for galstar output
	pixname=${infile%.in}
	statsfn="$pixname_$n.stats"
	binfn="${pixname}_$n_DM_Ar.dat"
	
	# Run galstar with the current pixel
	echo "$n of $maxpix: Running galstar ..."
	$galstardir/galstar $binfn:DM[5,20,120],Ar[0,10,400] --statsfile $statsfn --datafile $infile $n &> $outfn
	
	# Archive output, removing temporary files
	tar -rf $tarfn $binfn $statsfn
	rm $binfn $statsfn
done

# Add ASCII file containing std. out/err to tar archive and compress archive
tar -rf $tarfn $outerrfn.gz
gzip -9 $tarfn
rm $outerrfn.gz

cd $workingdir
echo "Done."
