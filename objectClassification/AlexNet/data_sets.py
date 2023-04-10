from torch.utils.data import Dataset
import data_utils as du
import os
import numpy as np
from PIL import Image

class TinyImageNet(Dataset):
    def __init__(self, path, transform = None) -> None:
        super(TinyImageNet, self).__init__()

        self.data  = []

        self.data_path = path
        self.transform = transform

        self.classes = tuple(du.valid_classes(self.data_path).keys())

        for index, name in enumerate(self.classes):
            files = os.listdir(os.path.join(self.data_path, name))
            self.data += list(zip(files, [index]*len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.data_path, self.classes[label])
        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label
    
if __name__ == '__main__':
    ## test data sets 
    data_path = './data/tiny-imagenet-200/train'
    classes = du.valid_classes(data_path)
    classes_list = tuple(classes.keys())
    
    ds = TinyImageNet(data_path)
    img, lbl = ds[1]
    
    print(f'Shape of the image {img.shape}')
    print(f'label:  {classes[classes_list[lbl]]}')

    w, h  = 126, 126

    img = Image.fromarray(img, 'RGB')
    img.show()