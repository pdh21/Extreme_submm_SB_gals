#!/usr/bin/env bash

for i in {2,3,4,5,6,7,8,9}; do
    papermill -p emulator_path '/Users/pdh21/Google_Drive/WORK/XID_plus/docs/notebooks/examples/SED_emulator/CIGALE_emulator_20210420_log10sfr_uniformAGN_z.npz' -p  source $i -p model 'full' XID+CIGALE-ESB_fit-SIDESsim.ipynb SIDESsim_$i.ipynb
    papermill -p emulator_path '/Users/pdh21/Google_Drive/WORK/XID_plus/docs/notebooks/examples/SED_emulator/CIGALE_emulator_20210420_log10sfr_uniformAGN_z.npz' -p  source $i -p model 'alt' XID+CIGALE-ESB_fit-SIDESsim.ipynb SIDESsim_alt_$i.ipynb
done