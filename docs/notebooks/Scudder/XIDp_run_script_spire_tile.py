import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import pickle
import dill
import sys
import os
import xidplus
import pymoc
import copy
from xidplus import moc_routines, catalogue
from xidplus import posterior_maps as postmaps
from builtins import input


try:
    taskid = np.int(os.environ['SGE_TASK_ID'])
    task_first=np.int(os.environ['SGE_TASK_FIRST'])
    task_last=np.int(os.environ['SGE_TASK_LAST'])

except KeyError:
    print("Error: could not read SGE_TASK_ID from environment")
    taskid = int(input("Please enter task id: "))
    print("you entered", taskid)


    
sources=Table.read('../../../data/Scudderetal2016/psw_sources.fits')
moc=pymoc.MOC()
moc.read('../../../data/Scudderetal2016/MOCS/MOC_{}.fits'.format(sources[taskid-1]['Fields']))

infile='./Master_prior.pkl'

with open(infile, 'rb') as f:
    obj=pickle.load(f)
priors=obj['priors']

for p in priors:
    p.moc=moc
    p.cut_down_prior()
    p.prior_bkg(0.0,5)
    p.get_pointing_matrix()
    p.upper_lim_map()

print('fitting '+ str(priors[0].nsrc)+' sources \n')
print('there are '+ str(priors[0].snpix)+' pixels')
print('fitting source:{} which is {} of {}'.format(sources[taskid-1]['Fields'],taskid-1,len(sources)))


from xidplus.numpyro_fit import SPIRE

fit=SPIRE.all_bands(priors,num_chains=4)

posterior=xidplus.posterior_numpyro(fit,priors)
output_folder='../../../data/Scudderetal2016/uniform_fit/
outfile=output_folder+'Source_{}'.format(sources[taskid-1]['Fields'])

xidplus.save(priors,posterior,outfile)
      
#post_rep_map=postmaps.replicated_maps(priors,posterior,nrep=2000)
#band=['psw','pmw','plw']
#for i,p in enumerate(priors):
#    Bayesian_Pval=postmaps.make_Bayesian_pval_maps(priors[i],post_rep_map[i])
#    wcs_temp=wcs.WCS(priors[i].imhdu)
#    ra,dec=wcs_temp.wcs_pix2world(priors[i].sx_pix,priors[i].sy_pix,0)
#    kept_pixels=np.array(moc_routines.sources_in_tile([tiles[taskid-1]],order,ra,dec))
#    Bayesian_Pval[np.invert(kept_pixels)]=np.nan
#    Bayes_map=postmaps.make_fits_image(priors[i],Bayesian_Pval)
#    Bayes_map.writeto(outfile+'_'+band[i]+'_Bayes_Pval.fits',overwrite=True)

#cat=catalogue.create_SPIRE_cat(posterior, priors[0],priors[1],priors[2])
#kept_sources=moc_routines.sources_in_tile([tiles[taskid-1]],order,priors[0].sra,priors[0].sdec)
#kept_sources=np.array(kept_sources)
#cat[1].data=cat[1].data[kept_sources]
#outfile=output_folder+'Tile_'+str(tiles[taskid-1])+'_'+str(order)

#cat.writeto(outfile+'_SPIRE_cat.fits',overwrite=True)

