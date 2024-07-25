import random
import xmltodict
import os
import shutil


@staticmethod
def split_list_idxs(list_len: int, split_ratio: float, seed: float = 2):
    """Given an input list, this method returns two lists of unique
           indices which correspond to the input list
        """
    random.seed(seed)
    idxs = list(range(list_len))
    num_el_A = int(split_ratio * list_len)
    idxs_a = random.sample(idxs, num_el_A)
    idxs_a.sort()
    idxs_b = list(set(idxs) - set(idxs_a))
    idxs_b.sort()
    return idxs_a, idxs_b


@staticmethod
def parse_xml(file_path):
    with open(file_path, 'r') as file:
        xml_content = file.read()
        xml_dict = xmltodict.parse(xml_content)
    return xml_dict


@staticmethod
def copy_content_to_folder(path_dir1: str, path_dir2: str, white_list: list = None):
    """Method which copies the contnet of folder1 to folder2.
        If folder2 does not exist, craetes folder2.
    """
    assert os.path.exists(path_dir1), f"Path {path_dir1} does not exist!"
    if not os.path.exists(path_dir2):
        os.makedirs(path_dir2)

    # Copy contents of the first folder
    for item in os.listdir(path_dir1):
        if white_list and item not in white_list:
            continue
        src_path = os.path.join(path_dir1, item)
        dest_path = os.path.join(path_dir2, item)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest_path)
