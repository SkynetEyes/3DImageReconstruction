import os

from classes.imagemisc import  ImageMisc

class Data:
    @classmethod
    def get_casa(cls, n: int):
        ROOT_DIR = '../SampleSet/MVS Data'

        # Allowed folders (based on what's in the directory)
        allowed_folders = {f"scan6_{i}_1" for i in range(2, 8)}
        allowed_folders.add("scan6_max_all")

        if n == 0:
            folder = os.path.join(ROOT_DIR, "scan6_max_all")
        else:
            folder = os.path.join(ROOT_DIR, f"scan6_{n}_1")

        # Ensure only valid folders are used
        if os.path.basename(folder) not in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Casa dataset")

        paths = ImageMisc.get_paths(folder, '*max.png')
        return paths


    @classmethod
    def get_chaleira(cls, n: int):
        ROOT_DIR = '../SampleSet/Chaleira'

        # Allowed dataset folders (no cl_3, cl_5, cl_7 etc.)
        allowed_folders = {"cl_2", "cl_4", "cl_6", "cl_8", "cl_all"}

        if n == 0:
            folder = os.path.join(ROOT_DIR, "cl_all")
        else:
            folder = os.path.join(ROOT_DIR, f"cl_{n}")

        # Ensure folder is valid
        if os.path.basename(folder) not in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Chaleira dataset")

        paths = ImageMisc.get_paths(folder, '*')
        return paths


    @classmethod
    def get_banana(cls, n: int):
        ROOT_DIR = '../SampleSet/Banana'

        # Allowed dataset folders
        allowed_folders = {f"ba_{i}" for i in range(2, 8)}
        allowed_folders.add("ba_all")

        if n == 0:
            folder = os.path.join(ROOT_DIR, "ba_all")
        else:
            folder = os.path.join(ROOT_DIR, f"ba_{n}")

        # Ensure the folder is valid (ignore "cal" or anything else)
        if os.path.basename(folder) not in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Banana dataset")

        paths = ImageMisc.get_paths(folder, '*')
        return paths


    @classmethod
    def get_banana2(cls, n: int):
        ROOT_DIR = '../SampleSet/Banana2'

        # Allowed folders (ignore calib_images and anything else)
        allowed_folders = {f"ba_{i}" for i in range(2, 11)} | {"ba_all"}

        if n == 0:
            folder = os.path.join(ROOT_DIR, "ba_all")
        else:
            folder = os.path.join(ROOT_DIR, f"ba_{n}")

        # Ensure only valid folders are used
        if not os.path.basename(folder) in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Banana dataset")

        paths = ImageMisc.get_paths(folder, '*')
        return paths

    @classmethod
    def get_banana3(cls, n: int):
        ROOT_DIR = '../SampleSet/Banana3'

        # Allowed folders (ignore calib_images and anything else)
        allowed_folders = {f"ba_{i}" for i in range(2, 10)} | {"ba_all"}

        if n == 0:
            folder = os.path.join(ROOT_DIR, "ba_all")
        else:
            folder = os.path.join(ROOT_DIR, f"ba_{n}")

        # Ensure only valid folders are used
        if not os.path.basename(folder) in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Banana dataset")

        paths = ImageMisc.get_paths(folder, '*')
        return paths

    @classmethod
    def get_cubo(cls, n: int):
        ROOT_DIR = '../SampleSet/Cubo'

        # Allowed folders (ignore calib_images and anything else)
        allowed_folders = {f"cu_{i}" for i in range(2, 10)} | {"cu_all"}

        if n == 0:
            folder = os.path.join(ROOT_DIR, "cu_all")
        else:
            folder = os.path.join(ROOT_DIR, f"cu_{n}")

        # Ensure only valid folders are used
        if not os.path.basename(folder) in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Cubo dataset")

        paths = ImageMisc.get_paths(folder, '*')
        return paths


    @classmethod
    def get_rosto(cls, n: int):
        ROOT_DIR = '../SampleSet/Rosto'

        # Allowed folders (ignore calib_images and anything else)
        allowed_folders = {f"ro_{i}" for i in range(2, 11)} | {"ro_all"}

        if n == 0:
            folder = os.path.join(ROOT_DIR, "ro_all")
        else:
            folder = os.path.join(ROOT_DIR, f"ro_{n}")

        # Ensure only valid folders are used
        if not os.path.basename(folder) in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Rosto dataset")

        paths = ImageMisc.get_paths(folder, '*')
        return paths


    @classmethod
    def get_cachorro(cls, n: int):
        ROOT_DIR = '../SampleSet/Cachorro'

        # Allowed folders (ignore calib_images and anything else)
        allowed_folders = {f"ca_{i}" for i in range(2, 10)} | {"ca_all"}

        if n == 0:
            folder = os.path.join(ROOT_DIR, "ca_all")
        else:
            folder = os.path.join(ROOT_DIR, f"ca_{n}")

        # Ensure only valid folders are used
        if not os.path.basename(folder) in allowed_folders:
            raise ValueError(f"Folder {folder} is not a valid Cachorro dataset")

        paths = ImageMisc.get_paths(folder, '*')
        return paths