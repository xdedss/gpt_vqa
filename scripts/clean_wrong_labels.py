# this is to clean rescuenet 
# where label of road / vehicle are in wrong order

src_path = 'rescuenet_with_wrong_label'
dst_path = 'rescuenet_with_label'

import os
import glob

import shutil
import re

def replace_text_in_file(file_path, pattern, replacement):
    with open(file_path, 'r') as file:
        filedata = file.read()
    
    filedata = re.sub(pattern, replacement, filedata)

    with open(file_path, 'w') as file:
        file.write(filedata)

def process_file(src_file, dst_file):
    # Copy the source file to the destination file
    shutil.copy(src_file, dst_file)

    # Perform the text replacements
    replace_text_in_file(dst_file, 'clear road', '[label1]')
    replace_text_in_file(dst_file, 'blocked road', 'clear road')
    replace_text_in_file(dst_file, 'vehicle', 'blocked road')
    replace_text_in_file(dst_file, '\\[label1\\]', 'vehicle')

os.makedirs(dst_path, exist_ok=True)

for src_file in glob.glob(os.path.join(src_path, "*.jsonl")):
    dst_file = os.path.join(dst_path, os.path.basename(src_file))
    process_file(src_file, dst_file)
