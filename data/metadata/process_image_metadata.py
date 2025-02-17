import os
import glob
import pandas as pd

from pathlib import Path

# Read the metadata CSV file and export dorsal and ventral image paths as rows into eval_image_paths.txt
def export_eval_image_paths(metadata_df, images_root=""):
    print("Export image paths in \n {}".format(metadata_df.head()))
    image_paths = {"image_paths":[]}
    image_missing = {"image_missing":[]}
    for index, row in metadata_df.iterrows():
        # Split the two image ids
        if ",," in row['imag_numbers']:
            dorsal, ventral = row['imag_numbers'].split(",,")
        elif "," in row['imag_numbers']:
            dorsal, ventral = row['imag_numbers'].split(",")
        elif "." in row['imag_numbers']:
            dorsal, ventral = row['imag_numbers'].split(".")
        name = "{} {}".format(row["genus"], row["species"])
        folder = row['image_folder'].replace('\\', '/')
        # Construct the image_path1 and image_path2
        image_path_d = os.path.join(images_root, name, folder, "IMG_{}.JPG".format(dorsal))
        image_path_v = os.path.join(images_root, name, folder, "IMG_{}.JPG".format(ventral))
        if os.path.isfile(image_path_d):
            image_paths["image_paths"].append(image_path_d)
        else:
            image_missing["image_missing"].append(image_path_d)
        if os.path.isfile(image_path_v):
            image_paths["image_paths"].append(image_path_v)
        else:
            image_missing["image_missing"].append(image_path_v)

    return  pd.DataFrame(image_paths),  pd.DataFrame(image_missing)

# Read the results.csv file and update the corresponding metadata CSV with dorsal and ventral measurement

"""
Purpose/Workflow 
- [Done] Read the metadata CSV/Excel file and produce a list of image files that needs to be processed by the Mothra 
- [TODO] Execute Mothra on the provided list of image file paths
- [TODO] Match and insert the results as a new column in the metadata CSV/Excel file for downstream assessment

"""
if __name__ == "__main__":
    images_root = "/home/rahul/workspace/data/UF_museum_data_2023"
    metadata_path = os.path.join(".", "231017_Battus_philenor_polydamas_FLMNH.xlsx")
    metadata_df = pd.read_excel(metadata_path, sheet_name = 0)
    image_paths_df, image_miss_df = export_eval_image_paths(metadata_df, images_root)
    # Write image paths to file
    image_paths_file="eval_image_paths.txt"
    image_paths_df.to_csv(image_paths_file, index=False)
    image_miss_file="eval_image_missing.txt"
    image_miss_df.to_csv(image_miss_file, index=False)
