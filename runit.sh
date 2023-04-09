#!/bin/bash
rm -rf ./build
rm tree.c
python setup.py build_ext --inplace