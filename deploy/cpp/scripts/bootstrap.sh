# download pre-compiled opencv lib
#OPENCV_URL=https://paddleseg.bj.bcebos.com/deploy/docker/opencv3gcc4.8.tar.bz2
#if [ ! -d "./deps/opencv3gcc4.8" ]; then
#    mkdir -p deps
#    cd deps
#    wget -c ${OPENCV_URL}
#    tar xvfj opencv3gcc4.8.tar.bz2
#    rm -rf opencv3gcc4.8.tar.bz2
#    cd ..
#fi
OPENCV_URL=https://bj.bcebos.com/paddleseg/deps/opencv346gcc4.8contrib.tar.bz2
if [ ! -d "./deps/opencv346gcc4.8contrib.tar.bz2" ]; then
    mkdir -p deps
    cd deps
    wget -c ${OPENCV_URL}
    tar xvfj opencv346gcc4.8contrib.tar.bz2
    #rm -rf opencv346gcc4.8contrib.tar.bz2
    cd ..
fi
