import numpy as np
import glob
import os
import shutil

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    orig_dir = "/Users/alexandracimpean/Desktop/VSC_Fairness/Nov2024/"
    target_dir = "/Users/alexandracimpean/Desktop/VSC_Fairness/Nov2024_core/"

    def copy(src, dest):
        for file_path in glob.glob(os.path.join(src, '**', 'pcn_log.csv'), recursive=True):
            print(file_path)
            new_path = file_path.replace(src, dest)
            new_dir = os.path.dirname(new_path)
            os.makedirs(new_dir, exist_ok=True)

            print("copy", file_path)
            print("to", new_path)
            shutil.copy(file_path, new_path)

    copy(orig_dir, target_dir)
