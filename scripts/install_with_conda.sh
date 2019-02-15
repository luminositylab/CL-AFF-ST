#!/bin/bash
echo "Creating environment for paper, named anlpe"
conda create -n anlpe python
echo "To use anlpe please use:"
echo "source activate anlpe"
source activate anlpe
echo "Installing AllenNLP inside paperenv"
if pip install allennlp ; then
	echo "AllenNLP environment setup successful!"
else
	echo "AllenNLP installation failed."
	exit
fi
echo "Installing the colored-traceback tool for debugging"
pip install colored-traceback
pip install pandas
source deactivate
echo "Installing rsub for remote file processing"
wget -O /usr/local/bin/rsub \https://raw.github.com/aurora/rmate/master/rmate
chmod a+x /usr/local/bin/rsub
echo "Creating GitHub directory"
mkdir github
echo "Please create personal directories in ~/github for development use."
