#!/bin/bash
echo "Updating and upgrading apt-get, installing dependencies"
apt-get update
apt-get upgrade
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
echo "Creating directory"
mkdir installation
cd installation
echo "Downloading miniconda3 installer"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
echo "Running miniconda3 installer"
chmod +x Miniconda3-latest-Linux-x86_64.sh
if ./Miniconda3-latest-Linux-x86_64.sh ; then
	echo "Miniconda3 installation successful!"
else
	echo "Miniconda3 installation failed."
	exit
fi
export PATH=~/miniconda3/bin:$PATH
cd ..
echo "Removing installer"
rm -r installation
#echo "Updating conda"
#conda update conda
echo "Creating environment for paper, named paperenv"
conda create -n paperenv python
echo "To use paperenv please use:"
echo "source activate paperenv"
source activate paperenv
echo "Installing AllenNLP inside paperenv"
if pip install allennlp ; then
	echo "AllenNLP environment setup successful!"
else
	echo "AllenNLP installation failed."
	exit
fi
echo "Installing the colored-traceback tool for debugging"
pip install colored-traceback
source deactivate
echo "Installing rsub for remote file processing"
wget -O /usr/local/bin/rsub \https://raw.github.com/aurora/rmate/master/rmate
chmod a+x /usr/local/bin/rsub
echo "Creating GitHub directory"
mkdir github
echo "Please create personal directories in ~/github for development use."