

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


cmake -DHPX_IGNORE_COMPILER_COMPATIBILITY=on -DCMAKE_CXX_COMPILER=mpic++ \
      -DTBBMALLOC_LIBRARY="$HOME/local/oneapi-tbb-2021.2.0/lib/intel64/gcc4.8/libtbbmalloc.so"           \
      -DTBBMALLOC_PROXY_LIBRARY="$HOME/local/oneapi-tbb-2021.2.0/lib/intel64/gcc4.8/libtbbmalloc_proxy.so"           \
      -DCMAKE_BUILD_TYPE=$1                                                                                                                            \
      -DCOSMICTIGER_WITH_FMM_ORDER=7  \
      ..

