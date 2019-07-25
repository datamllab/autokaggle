import os
import tempfile
import string
import random
import json


def generate_rand_string(size):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))
    
def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def temp_path_generator():
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokaggle')
    return path


def rand_temp_folder_generator():
    """Create and return a temporary directory with the path name '/temp_dir_name/autokeras' (E:g:- /tmp/autokeras)."""
    sys_temp = temp_path_generator()
    path = sys_temp + '_' + generate_rand_string(6)
    ensure_dir(path)
    return path

def write_json(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
        
def read_json(filename):
    with open(filename, 'rb') as infile:
        return json.load(infile)

def write_csv(filename, line):
    with open(filename, "a") as f:
        f.write(", ".join(map(str, line)))
        f.write("\n")
