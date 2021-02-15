#!/bin/bash
source activate new
for source in {0..33}
do
export source
jupyter nbconvert --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1 --execute XID+IR_SED-ESB.ipynb
done
exit 0
