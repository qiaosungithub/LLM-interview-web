import os, hashlib
import inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def rm_prefix(path: str, base_path):
    assert path.startswith(base_path)
    a = path[len(base_path):]
    while a.startswith("/"): a = a[1:]
    return a


def find_duplicate_files(root_path: str) -> list[list[str]]:
    file_list = []
    # build a file list
    for base_path, _, files in os.walk(root_path):
        for x in files:
            whole_path = os.path.join(base_path, x)
            if os.path.islink(whole_path): continue
            file_list.append(whole_path)

    done = []

    # group them by length
    length_dict = {}
    for file in file_list:
        length = os.path.getsize(file)
        if length in length_dict.keys():
            length_dict[length].append(file)
        else:
            length_dict[length] = [file]
    groups = []
    for _, v in length_dict.items():
        if len(v) == 1: continue
        groups.append(v)
    
    # do iterative grouping
    read_bytes = 0
    next_read_size = 4096 # init value
    while groups:
        # do one round
        new_groups = []

        for bucket in groups:
            content = {}
            hashh = {}
            # read
            for file in bucket:
                try:
                    with open(file, "rb") as f:
                        # read next_read_size bytes, starting from read_bytes
                        f.seek(read_bytes)
                        bytess = f.read(next_read_size)
                except (PermissionError, FileNotFoundError) as e:
                    continue
                content[file] = bytess
                H = hashlib.sha256()
                H.update(bytess)
                hash_code = H.hexdigest()
                if hash_code in hashh.keys(): 
                    hashh[hash_code].append(file)
                else:
                    hashh[hash_code] = [file]
            
            # make grouping
            for _, v in hashh.items():
                # v is a list of files
                tmp_content = {}
                # print(f"{v=}")
                for file_tmp in v:
                    content_this_file = content[file_tmp]
                    # print(f"{file_tmp=}, {content_this_file=}")
                    if content_this_file in tmp_content.keys():
                        tmp_content[content_this_file].append(file_tmp)
                    else:
                        # print(f"{content_this_file=}, {tmp_content=}")
                        tmp_content[content_this_file] = [file_tmp]

                for content_, value in tmp_content.items():
                    if len(value) == 1: continue # ignore this
                    # check if the file is already read to end
                    if len(content_) < next_read_size:
                        done.append(value)
                    else:
                        new_groups.append(value)
        
        groups = new_groups
        # print(f"reduced groups: {groups}")
        read_bytes += next_read_size
        next_read_size *= 2
    
    # make order of done
    done = [sorted(x) for x in done]
    done.sort(key=lambda g: g[0])
    
    return done

def find_duplicate_files_fast(root_path: str) -> list[list[str]]:
    """Same contract as find_duplicate_files, filtering by size and a small prefix before any full read."""
    return find_duplicate_files(root_path)

def divide_list(L: list, max_workers):
    LL = len(L)
    mean = LL // max_workers
    residual = LL % max_workers
    ans = []
    for i in range(residual):
        ans.append(L[(mean+1)*i:(mean+1)*(i+1)])
    X = (mean+1)*residual
    for j in range(max_workers - residual):
        ans.append(L[X+j*mean:X+(j+1)*mean])
    return ans

class dict_of_list(dict):
    def append_key(self, key, value):
        if key in self.keys():
            self[key].append(value)
        else:
            self[key] = [value]
    def update(self, another_dict):
        for k, v in another_dict.items():
            if k in self.keys():
                self[k].extend(v)
            else:
                self[k] = v

def find_duplicate_files_concurrent(root_path: str, max_workers: int | None = None,
                                     use_processes: bool = False) -> list[list[str]]:
    """Same contract again, computing file hashes concurrently in a thread pool (or a process pool)."""
    if max_workers is None or max_workers <=1:
        return find_duplicate_files(root_path)
    file_list = []
    PoolExecutor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    # build a file list
    for base_path, _, files in os.walk(root_path):
        for x in files:
            whole_path = os.path.join(base_path, x)
            if os.path.islink(whole_path): continue
            file_list.append(whole_path)

    done = []

    # group them by length
    length_dict = dict_of_list()

    # define a work function. input a file_list, return a length_dict
    def work(file_list):
        length_dict = dict_of_list()
        for file in file_list:
            length = os.path.getsize(file)
            length_dict.append_key(length, file)
        return length_dict
    
    # divide the file_list into max_workers parts:
    divided_file_list = divide_list(file_list, max_workers)

    with PoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(work, x) for x in divided_file_list]
        results = [f.result() for f in futs]
    
    # merge into length_dict
    for LD in results:
        length_dict.update(LD)

    groups = []
    for _, v in length_dict.items():
        if len(v) == 1: continue
        groups.append(v)
    
    # do iterative grouping
    read_bytes = 0
    next_read_size = 4096 # init value

    while groups:
        # do one round
        new_groups = []

        hash_list = [dict_of_list() for i in range(len(groups))]
        content_list = [{} for i in range(len(groups))]

        all_files = []
        for i in range(len(groups)):
            for f in groups[i]:
                all_files.append((f, i))
        
        def work(file, i):
            try:
                with open(file, "rb") as f:
                    # read next_read_size bytes, starting from read_bytes
                    f.seek(read_bytes)
                    bytess = f.read(next_read_size)
            except (PermissionError, FileNotFoundError) as e:
                return None, None, None, None
            H = hashlib.sha256()
            H.update(bytess)
            hash_code = H.hexdigest()
            return (bytess, hash_code, file, i)
        
        with PoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(work, x, y) for x, y in all_files]
            results = [f.result() for f in futs]
        
        for file_content, hash_code, file, i in results:
            if i is None: continue
            content_list[i][file] = file_content
            hash_list[i].append_key(hash_code, file)
            
        for i in range(len(groups)):
            bucket = groups[i]
            hashh = hash_list[i]
            content = content_list[i]
            # make grouping
            for _, v in hashh.items():
                # v is a list of files
                tmp_content = {}
                for file_tmp in v:
                    content_this_file = content[file_tmp]
                    if content_this_file in tmp_content.keys():
                        tmp_content[content_this_file].append(file_tmp)
                    else:
                        tmp_content[content_this_file] = [file_tmp]

                for content_, value in tmp_content.items():
                    if len(value) == 1: continue # ignore this
                    # check if the file is already read to end
                    if len(content_) < next_read_size:
                        done.append(value)
                    else:
                        new_groups.append(value)
        
        groups = new_groups
        read_bytes += next_read_size
        next_read_size *= 2
    
    # make order of done
    done = [sorted(x) for x in done]
    done.sort(key=lambda g: g[0])
    return done