import os
import shutil

def clear_pycache(directory):
    """
    Recursively delete all __pycache__ directories in the given directory.
    
    Parameters:
    - directory (str): The root directory to start the search. Defaults to the current directory.
    """
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                print(f"Deleting: {pycache_path}")
                shutil.rmtree(pycache_path)

if __name__ == "__main__":
    # Specify the directory to clear __pycache__ (default is the current directory)
    clear_pycache("C:\\EL-3rdsem\\STRESS\\ThirdAttempt\\webapp")
