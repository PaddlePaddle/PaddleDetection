DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# Extract the data.
echo "Extracting..."
unzip train2014.zip
unzip val2014.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2014.zip
unzip annotations_trainval2017.zip

