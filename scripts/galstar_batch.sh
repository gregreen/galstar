#!/bin/bash

# Create an empty tar archive to store the output
tarfn=`readlink -m ${TAROUT%/}`
tar --create --file=$tarfn --files-from=/dev/null
echo "Writing binned pdfs and statistics to $tarfn."

# Determine output file for std. output/error
outerrfn=`readlink -m $OUTERR`
outerrdir=`readlink -m ${outerrfn%/*}`
outerrfnshort="${OUTERR##*/}"
echo "Writing std. out/err to $outerrfn."

# Determine working directory
workingdir=`pwd`
tmpdir=`readlink -m ${TMPDIR%/}`

# Give each input file to galstar
nfiles=`ls -l *.in | wc -l`
counter=1
for f in *.in; do
	# Determine filenames for galstar output
	pixname=${f%.in}
	statsfn="$pixname.stats"
	binfn="${pixname}_DM_Ar.dat"
	
	# Run galstar with the current l.o.s input file
	echo "$counter of $nfiles: Running galstar with $f ..."
	${GALSTARDIR%/}/galstar $tmpdir/$binfn:DM[5,20,120],Ar[0,10,400] --statsfile $tmpdir/$statsfn --datafile $f &>> $outerrfn
	
	# Compress and archive output, removing temporary files
	cd $tmpdir	# Move to temporary directory, so that the tarball has a flat directory structure
	gzip -9 $binfn
	tar -rf $tarfn $binfn.gz $statsfn
	rm $binfn.gz $statsfn
	cd $workingdir	# Return to the working directory
	counter=`expr $counter + 1`
done

# Add ASCII file containing std. out/err to tar archive
cd $outerrdir
gzip -9 $outerrfn
tar -rf $tarfn $outerrfnshort.gz
rm $outerrfnshort.gz
cd $workingdir

echo "Done."
