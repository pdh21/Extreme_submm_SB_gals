from astropy.io import ascii, fits
from astropy import wcs


import numpy as np
import xidplus
from xidplus import moc_routines
import pickle

from xidplus import sed
SEDs, df=sed.berta_templates(PACS=False)
priors,posterior=xidplus.load(filename='../priors/prior_XID+IR_SED_ESB_32_added_source_flag_gaia.pkl')
import xidplus.stan_fit.SED as SPM
temps=[15,20,21,9,30,1,5,14,0]
print(SEDs.shape)
priors=[priors[0],priors[3],priors[4],priors[5]]
fit=SPM.MIPS_SPIRE(priors,SEDs[temps,:,:],chains=4,iter=750,max_treedepth=12,seed=2911,adapt_delta=0.96)
posterior=sed.posterior_sed(fit,priors,SEDs)
xidplus.save(priors, posterior, 'XID+MIPS_SPIRE_SED_ESB_32_750iter_flag_gaia_wdiv_adaptdelta96_uniform_flux_uniformlir')
