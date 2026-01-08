#!/bin/bash


pushd "mc_pna/"
python3 ../prepare.py -t input.txt --method char
popd

mv mc_pna ../

pushd "mc_out/"
python3 ../prepare.py -t input.txt --method char
popd

mv mc_out ../

pushd "mc_ga/"
python3 ../prepare.py -t input.txt --method char
popd

mv "mc_ga/" ../
