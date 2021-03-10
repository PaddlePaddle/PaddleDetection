# All rights `PaddleDetection` reserved
# References:
#   @inproceedings{yang2016wider,
#   Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
#   Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#   Title = {WIDER FACE: A Face Detection Benchmark},
#   Year = {2016}}

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget https://dataset.bj.bcebos.com/wider_face/WIDER_train.zip
wget https://dataset.bj.bcebos.com/wider_face/WIDER_val.zip
wget https://dataset.bj.bcebos.com/wider_face/wider_face_split.zip
# Extract the data.
echo "Extracting..."
unzip -q WIDER_train.zip
unzip -q WIDER_val.zip
unzip -q wider_face_split.zip
