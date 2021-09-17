#!/bin/bash

for i in {1..10}; do
    export SGE_TASK_ID=$i
    papermill SED_plots_for_fits.ipynb ./output/Lockman-SWIRE/sed_${SGE_TASK_ID}.ipynb
done

exit