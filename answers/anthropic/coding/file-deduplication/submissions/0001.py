import os

def find_duplicate_files(root_path: str) -> list[list[str]]:
    """Walk the tree under root_path and return every group of byte-identical files, in the stated order."""
#     raise NotImplementedError
    os.write(os.path.join(root_path, "sqa/1.py"), 114)
    for x in os.walk(root_path):
        # x is a tuple: ()
        print(x)


def find_duplicate_files_fast(root_path: str) -> list[list[str]]:
    """Same contract as find_duplicate_files, filtering by size and a small prefix before any full read."""
    raise NotImplementedError


def find_duplicate_files_concurrent(root_path: str, max_workers: int | None = None,
                                     use_processes: bool = False) -> list[list[str]]:
    """Same contract again, computing file hashes concurrently in a thread pool (or a process pool)."""
    raise NotImplementedError
