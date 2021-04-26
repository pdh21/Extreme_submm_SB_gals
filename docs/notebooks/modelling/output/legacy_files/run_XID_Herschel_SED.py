from astropy.io import ascii, fits
from astropy import wcs


import numpy as np
import xidplus
from xidplus import moc_routines
import pickle

from xidplus import sed
SEDs, df=sed.berta_templates(MIPS=False)
priors,posterior=xidplus.load(filename='../priors/prior_XID+Herschel_SED_ESB_32_added_source.pkl')
import xidplus.stan_fit.SED as SPM
temps=[15,20,21,9,30,1,5,14,0]
bands=[0,1,2,4,5]
print(SEDs.shape)
fit=SPM.PACS_SPIRE(priors,SEDs[temps,:,:],chains=2,iter=500,max_treedepth=12)
posterior=sed.posterior_sed(fit,priors,SEDs)
xidplus.save(priors, posterior, 'XID+Herschel_SED_ESB_32_added_source')