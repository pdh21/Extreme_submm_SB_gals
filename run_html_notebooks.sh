#!/bin/bash

for source in {1..30}
do
jupyter nbconvert --to html --execute Source_investigation.ipynb
done
exit 0