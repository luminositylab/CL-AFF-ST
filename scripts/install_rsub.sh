#!/bin/bash
echo "Installing rsub for remote file processing"
wget -O /usr/local/bin/rsub \https://raw.github.com/aurora/rmate/master/rmate
chmod a+x /usr/local/bin/rsub
echo "Creating GitHub directory"
mkdir github
echo "Please create personal directories in ~/github for development use."
