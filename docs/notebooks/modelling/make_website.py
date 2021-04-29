from mdtable import MDTable
from mdutils.mdutils import MdUtils
import os
import glob
mdFile = MdUtils(file_name='fits',title='Fits')
mdFile.new_paragraph("The fits for the different fields can be found at the following locations:")

field=['Lockman-SWIRE','ELAIS-S1','XMM-LSS']
for i in field:
    srcs=glob.glob('./output/{}/*.ipynb'.format(i))
    #for s in range(1,len(srcs)+1):
        #os.system('jupyter nbconvert ./output/{}/fit_{}.ipynb --to markdown'.format(i,s))
    mdFile.new_header(level=1, title=i)
    mdFile.write('  \n')
    markdown=MDTable('../../../data/MRR2018_tables/{}_web.csv'.format(i))
    markdown_string_table = markdown.get_table()
    mdFile.write(markdown_string_table)
    mdFile.write('  \n')
mdFile.create_md_file()
