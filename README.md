# rmhd
Reduced MHD GPU-based Simulation Code.

This is based on the Gandalf simulation code written by Anjor Kanekar and William Dorland.
This version is maintained by Ian Abel, with contributinos from Dan Martin and Nicholas Bidler.

To build and run on Perlmutter.nersc.gov:

To build and run an example on Perlmutter.nersc.gov:

1. clone from github:

git clone git@github.com:ianabel/rmhd.git

2. set modules:

source setupModules-Perlmutter

3. build code:

make

4. Run test code on login node:

cd test

../gandalf a01

5. run on gpu nodes (from rmhd/test directory):

salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu

../gandalf a01



