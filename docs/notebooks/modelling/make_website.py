from mdtable import MDTable
from mdutils.mdutils import MdUtils

mdFile = MdUtils(file_name='fits',title='Fits')
mdFile.new_paragraph("The fits for the different fields can be found at the following locations:")

field=['Lockman-SWIRE','ELAIS-S1','XMM-LSS']
for i in field:
    print(i)
    mdFile.new_header(level=1, title=i)
    markdown=MDTable('../../../data/MRR2018_tables/{}_sources.csv'.format(i))
    markdown_string_table = markdown.get_table()
    mdFile.write(markdown_string_table)
    mdFile.new_paragraph()
mdFile.create_md_file()
