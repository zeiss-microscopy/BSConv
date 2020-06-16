import os.path
import tarfile

import torchvision.datasets
import torchvision.datasets.folder
import torchvision.datasets.utils


class StanfordDogs(torchvision.datasets.VisionDataset):
    """
    Dataset class for the StanfordDogs (aka ImageNetDogs) dataset
    (http://vision.stanford.edu/aditya86/ImageNetDogs/).
    
    This class is PyTorch/torchvision compatible.
    
    The `root` dir must contain the raw files `images.tar` and `lists.tar`,
    which are available from the URL above. They can also be downloaded
    automatically by setting `download=True`.
    """
    
    sources = (
        {
            "url": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            "md5": "1bb1f2a596ae7057f99d7d75860002ef",
            "filename": "images.tar",
            "extracted_filenames": ("Images",),
        },
        {
            "url": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
            "md5": "edbb9f16854ec66506b5f09b583e0656",
            "filename": "lists.tar",
            "extracted_filenames": ("file_list.mat", "test_list.mat", "train_list.mat"),
        },
    )
    
    def __init__(self, root, train=True, transforms=None, transform=None, target_transform=None, download=False, loader=torchvision.datasets.folder.default_loader):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.loader = loader
        
        self.train = train
        
        if download:
            self.download()
            
        self.unique_class_names = self.read_unique_class_names()
        self.image_filenames = self.read_image_filenames()
        self.targets = tuple(self.get_class_index_from_image_filename(image_filename) for image_filename in self.image_filenames)
        
    @staticmethod
    def read_file_list_from_mat(filename):
        """
        Reads a file list from a .mat file as formatted in this dataset.
        Requires SciPy.
        """
        import scipy.io
        mat = scipy.io.loadmat(filename)
        return [value[0] for value in mat["file_list"][:, 0]]
    
    def read_unique_class_names(self):
        """
        Load the class names from the file list and return all unique values
        as a tuple.
        """
        image_filenames = self.read_file_list_from_mat(filename=os.path.join(self.root, "file_list.mat"))
        class_names = set()
        for image_filename in image_filenames:
            class_name = self.get_class_name_from_image_filename(image_filename=image_filename)
            class_names.add(class_name)
        return tuple(sorted(class_names))
        
    def get_class_name_from_image_filename(self, image_filename):
        """
        Derive the class name from the given image filename.
        """
        return os.path.basename(image_filename).split("_")[0]
    
    def get_class_index_from_image_filename(self, image_filename):
        class_name = self.get_class_name_from_image_filename(image_filename=image_filename)
        return self.unique_class_names.index(class_name)
    
    def read_image_filenames(self):
        """
        Read all image filenames from the dataset's list files. Varies whether
        `self.train` is `True` or `False`.
        """
        if self.train:
            list_filename = "train_list.mat"
            image_count = 12000
        else:
            list_filename = "test_list.mat"
            image_count = 8580
        image_filenames = self.read_file_list_from_mat(filename=os.path.join(self.root, list_filename))
        image_filenames = tuple(os.path.join(self.root, "Images", image_filename) for image_filename in image_filenames)

        assert len(image_filenames) == image_count

        return image_filenames
    
    def __getitem__(self, index):
        """
        Return the `index`-th sample of the dataset, which is a tuple
        `(image, target)`.
        """
        image = self.loader(self.image_filenames[index])
        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return (image, target)
    
    def __len__(self):
        """
        Returns the number of images in this dataset. Varies whether
        `self.train` is `True` or `False`.
        """
        return len(self.image_filenames)
    
    def download(self):
        """
        Download and extract the neccessary source files into the root
        directory.
        """
        for source in self.sources:
            full_filename = os.path.join(self.root, source["filename"])
            
            # download
            torchvision.datasets.utils.download_url(url=source["url"], root=self.root, filename=source["filename"], md5=source["md5"])
                
            # extract
            if not all(os.path.exists(os.path.join(self.root, extracted_filename)) for extracted_filename in source["extracted_filenames"]):
                print("Extracting '{}' to '{}'".format(source["filename"], self.root))
                with tarfile.open(full_filename, "r") as tar:
                    tar.extractall(path=self.root)
            else:
                print("File '{}' was already extracted, skipped extraction".format(source["filename"]))
