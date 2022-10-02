#!/bin/sh

WORKdir=$PWD
echo " --- ${WORKdir} --- "

cd ${WORKdir}/webapp

docker build -t dogbreedsclassification .

docker run -p 5000:5000 dogbreedsclassification
