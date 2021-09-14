for i in {100..142}
do
   echo "view $i"
   ./healpix2silo --in=view.$i.dat --out=view.$i.silo
done
