import os
import glob
import pandas as pd

from pathlib import Path

# Read the metadata CSV file and export dorsal and ventral image paths as rows into eval_image_paths.txt
def export_eval_image_paths(metadata_df, output_file="eval_image_paths.txt"):
    print("Export image paths in \n {}".format(metadata_df.head()))
    image_paths = {"image_paths":[]}
    for index, row in metadata_df.iterrows():
        # Split the two image ids
        if ",," in row['imag_numbers']:
            dorsal, ventral = row['imag_numbers'].split(",,")
        elif "," in row['imag_numbers']:
            dorsal, ventral = row['imag_numbers'].split(",")
        elif "." in row['imag_numbers']:
            dorsal, ventral = row['imag_numbers'].split(".")
        # Construct the image_path1 and image_path2
        image_paths["image_paths"].append(row['image_folder']+dorsal)
        image_paths["image_paths"].append(row['image_folder']+ventral)
    image_paths_df = pd.DataFrame(image_paths)
    image_paths_df.to_csv(output_file, index=False)
    return 

# Read the results.csv file and update the corresponding metadata CSV with dorsal and ventral measurement

"""
Purpose/Workflow 
- [Done] Read the metadata CSV/Excel file and produce a list of image files that needs to be processed by the Mothra 
- [TODO] Execute Mothra on the provided list of image file paths
- [TODO] Match and insert the results as a new column in the metadata CSV/Excel file for downstream assessment

"""
if __name__ == "__main__":
    
    metadata_path = os.path.join(".", "231017_Battus_philenor_polydamas_FLMNH.xlsx")
    metadata_df = pd.read_excel(metadata_path, sheet_name = 0)
    export_eval_image_paths(metadata_df)
