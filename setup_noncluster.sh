# set up environment

if [[ ! -d RAJA-install ]]; then
  git clone --recursive https://github.com/LLNL/RAJA.git
  mkdir RAJA-build
  mkdir RAJA-install
  echo 'Success in environment set up'

  # build RAJA 
  cd RAJA-build
  cmake -DRAJA_ENABLE_TESTS=Off -DRAJA_ENABLE_EXAMPLES=Off -DRAJA_ENABLE_EXERCISES=Off -DCMAKE_INSTALL_PREFIX=../RAJA-install -DENABLE_OPENMP=On ../RAJA
  make -j && make install
  echo 'Success in build'

  cd ../
fi

# make and run program
cd ./example
make
cd ..
