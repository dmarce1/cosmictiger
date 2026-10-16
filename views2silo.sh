for i in {100..142}
do
   echo "view $i"
   ./healpix2silo --input.file=view.$i.dat --output.file=view.$i.silo
done
