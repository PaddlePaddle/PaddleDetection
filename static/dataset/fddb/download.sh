# All rights `PaddleDetection` reserved
# References:
#   @TechReport{fddbTech,
#      author = {Vidit Jain and Erik Learned-Miller},
#      title =  {FDDB: A Benchmark for Face Detection in Unconstrained Settings},
#      institution =  {University of Massachusetts, Amherst},
#      year = {2010},
#      number = {UM-CS-2010-009}
#   }

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
# external link to the Faces in the Wild dataset and annotations file
wget http://tamaraberg.com/faceDataset/originalPics.tar.gz
wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz
wget http://vis-www.cs.umass.edu/fddb/evaluation.tgz

# Extract the data.
echo "Extracting..."
tar -zxf originalPics.tar.gz
tar -zxf FDDB-folds.tgz
tar -zxf evaluation.tgz

# Generate full image path list and groundtruth in FDDB-folds:
cd FDDB-folds
cat `ls|grep -v"ellipse"` > filePath.txt && cat *ellipse* > fddb_annotFile.txt
cd ..
echo "-------------   All done!   --------------"
