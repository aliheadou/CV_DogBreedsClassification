#!/bin/sh

WORKdir=$PWD
echo " --- ${WORKdir} --- "

cd ${WORKdir}/webapp

docker pull aliheadou/dogbreedsclassification:v0

docker run -p 5000:5000 aliheadou/dogbreedsclassification:v0
