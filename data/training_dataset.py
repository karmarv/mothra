import os 
import glob
import numpy as np
from pathlib import Path





import os
def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)

def list_leaf_dirs(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      if not dirs:
         print("Leaf: {}".format(root))
   return(paths)

folder = "../UF_museum_data_2023"
ext    = ".JPG"

list_leaf_dirs(folder, ext)