for i in {0..100}
do
   echo "view $i"
   ./healpix2silo --in=view.$i.dat --out=view.$i.silo
done
