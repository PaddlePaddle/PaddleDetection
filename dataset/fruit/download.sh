DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget https://dataset.bj.bcebos.com/PaddleDetection_demo/fruit-detection.tar
# Extract the data.
echo "Extracting..."
tar xvf fruit-detection.tar
rm -rf fruit-detection.tar
