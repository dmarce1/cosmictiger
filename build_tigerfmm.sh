

set -x

source ~/scripts/sourceme.sh gperftools
source ~/scripts/sourceme.sh hwloc
source ~/scripts/sourceme.sh vc
source ~/scripts/sourceme.sh silo
source ~/scripts/sourceme.sh $1/hpx

#rm -rf $1
#mkdir $1
cd $1
rm CMakeCache.txt
rm -r CMakeFiles


cmake -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc \
      -DTBBMALLOC_LIBRARY="$HOME/local/oneapi-tbb-2021.2.0/lib/intel64/gcc4.8/libtbbmalloc.so"           \
      -DTBBMALLOC_PROXY_LIBRARY="$HOME/local/oneapi-tbb-2021.2.0/lib/intel64/gcc4.8/libtbbmalloc_proxy.so"           \
      -DCMAKE_BUILD_TYPE=$1                                                                                                                            \
      ..

