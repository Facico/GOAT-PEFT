set -xe
export KAGGLE_USERNAME=xxx
export KAGGLE_KEY=yyy

sudo apt install unzip
pip install kaggle
# stanford cars dataset (ref: https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616)
mkdir stanford_cars
cd stanford_cars
echo "downloading kaggle dataset, if not success, try to download it manually into ./dataset/~data/"
kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars
unzip torchvision-stanford-cars.zip 

# ressic45
mkdir resisc45
cd resisc45
# (manual download) 
# wget https://sov8mq.dm.files.1drv.com/y4m4JOwqH0JXOhy8Y8QgSbFce3DlJKhuMVclXqqhqAPeUMnn1oEZ1O3seAcXqhzrrwFny_Oo3NuyLN_baLTq-eVEyhipspKOBrUrPIv3qr0DZTPvrq4SYCFInkim-_j4wa3IB6RKdSPPJ8Dhzyo8ZDRYD2xMpVVnJnhnqoEG8eG9fcBag1-dvcl9GkCF4XSlDnneyxiPNVH2uDcwCo0H_n2zA -O NWPU-RESISC45.rar
# if [ ! -f NWPU-RESISC45.rar ]; then
#     echo 'Please manually download the dataset from:'
#     echo 'https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp'
#     echo 'Once downloaded, press [Enter] to continue...'
#     read -r  # 等待用户按下 Enter 键
# fi
# sudo apt -y install unar
# unar NWPU-RESISC45.rar
# wget -O resisc45-train.txt "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt"
# wget -O resisc45-val.txt "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt"
# wget -O resisc45-test.txt "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt"
# rm -rf NWPU-RESISC45.rar
huggingface-cli download --repo-type dataset torchgeo/resisc45 --local-dir .
unzip NWPU-RESISC45.zip
# huggingface-cli download --repo-type dataset torchgeo/resisc45 resisc45-train.txt --local-dir .
# huggingface-cli download --repo-type dataset torchgeo/resisc45 resisc45-val.txt --local-dir .
# huggingface-cli download --repo-type dataset torchgeo/resisc45 resisc45-test.txt --local-dir .
cd ..

# dtd
mkdir dtd
cd dtd
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
rm -rf dtd-r1.0.1.tar.gz
mv dtd/images images
mv dtd/imdb/ imdb
mv dtd/labels labels
cat labels/train1.txt labels/val1.txt > labels/train.txt
cat labels/test1.txt > labels/test.txt
cd ..
python process_dtd.py

# euro_sat
mkdir euro_sat
cd euro_sat
wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip
rm -rf EuroSAT.zip
cd ..
python process_euro.py

# sun397
mkdir sun397
cd sun397
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
unzip Partitions.zip
tar -xvzf SUN397.tar.gz
rm -rf SUN397.tar.gz
cd ..
python process_sun.py
