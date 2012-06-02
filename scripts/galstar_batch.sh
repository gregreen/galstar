#!/bin/bash

# Create an empty tar archive to store the output
tarfn=$OUTDIR/$TAROUT
tar --create --file=$tarfn --files-from=/dev/null

# Give each input file to galstar
for f in *.in; do
	# Determine output filenames
	pixname=${f%.in}
	statsfn="$OUTDIR/$pixname.stats"
	binfn="$OUTDIR/${pixname}_DM_Ar.dat"
	
	# Run galstar on input
	echo "Running galstar with $f ..."
	$GALSTARDIR/galstar $binfn:DM[5,20,120],Ar[0,10,400] --statsfile $statsfn --datafile $f &> $OUTDIR/out.txt
	
	# Compress and archive output
	gzip -9 $binfn
	tar -rf $tarfn $binfn.gz $statsfn
	rm $binfn $statsfn
done

echo "Done."
