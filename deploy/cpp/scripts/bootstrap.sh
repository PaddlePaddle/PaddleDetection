# download pre-compiled opencv lib
OPENCV_URL=https://paddleseg.bj.bcebos.com/deploy/deps/opencv346.tar.bz2
if [ ! -d "./deps/opencv346" ]; then
    mkdir -p deps
    cd deps
    wget -c ${OPENCV_URL}
    tar xvfj opencv346.tar.bz2
    rm -rf opencv346.tar.bz2
    cd ..
fi

