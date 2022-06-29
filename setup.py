import distutils
from distutils.core import setup
import glob

bin_files = glob.glob("bin/*.py")

# The main call
setup(name='despyPIFF',
      version ='3.0.7',
      license = "GPL",
      description = "DESDM tools for PIFF",
      author = "Robert Gruendl",
      author_email = "gruendl@illinois.edu",
      packages = ['despyPIFF'],
      package_dir = {'': 'python'},
      scripts = bin_files,
      data_files=[('ups',['ups/despyPIFF.table'])],
      )
