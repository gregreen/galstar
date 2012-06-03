#!/bin/bash
#
# Runs galstar on all input files in the working directory,
# archiving the results in a tarball.
#
# Requires the following environmental variables to be set:
#    TARIN      = tarball containing inputs
#    TMPDIR     = temporary directory
#    GALSTARDIR = directory containing galstar executable
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

# Determine filename for std. output/error
outerrfn="$tmpdir/out-err.txt"

# Get list of .in files in the input tarball
tarin=`readlink -m $TARIN`
infilelist=`tar -tf $TARIN | grep .in`

# Determine number of pixels in tarball
npix=0
for infile in $infilelist; do
        npix=`expr $npix + 1`
done

cd $tmpdir

# Give each input file to galstar
counter=1
for infile in $infilelist; do
	# Determine filenames for galstar output
	pixname=${infile%.in}
	statsfn="$pixname.stats"
	binfn="${pixname}_DM_Ar.dat"
	
	# Extract the current input file from the tarball
	tar -xf $tarin $infile
	
	# Run galstar with the current l.o.s input file
	echo "$counter of $npix: Running galstar with $infile ..."
	$galstardir/galstar $tmpdir/$binfn:DM[5,20,120],Ar[0,10,400] --statsfile $tmpdir/$statsfn --datafile $tmpdir/$infile &>> $outerrfn
	
	# Remove the used input file
	rm $infile
	
	# Compress and archive output, removing temporary files
	#cd $tmpdir	# Move to temporary directory, so that the tarball has a flat directory structure
	gzip -9 $binfn
	tar -rf $tarfn $binfn.gz $statsfn
	rm $binfn.gz $statsfn
	#cd $workingdir	# Return to the working directory
	counter=`expr $counter + 1`
done

# Add ASCII file containing std. out/err to tar archive
#cd $tmpdir
gzip -9 $outerrfn
tar -rf $tarfn $outerrfn.gz
rm $outerrfn.gz

cd $workingdir
echo "Done."
