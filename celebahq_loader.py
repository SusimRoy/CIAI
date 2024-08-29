import data
from torch.utils.data import DataLoader


def load_data():
        data_dir = 'CelebA-HQ'
        attribute = 'Male'
        val_transform = data.get_transform('celebahq', 'imval')
        clean_dset1 = data.get_dataset('celebahq', 'train', attribute, root=data_dir, transform=val_transform,
                                      fraction=100, data_seed=0, return_path=True) 
        clean_dset2 = data.get_dataset('celebahq', 'val', attribute, root=data_dir, transform=val_transform,
                                      fraction=100, data_seed=0, return_path=True) 
        clean_dset3 = data.get_dataset('celebahq', 'test', attribute, root=data_dir, transform=val_transform,
                                      fraction=100, data_seed=0, return_path=True) 
        # clean_dset = data.get_dataset('celebahq', 'train', attribute, root=data_dir, transform=val_transform,
        #                               fraction=100, data_seed=0) # data_seed randomizes here
        dset = clean_dset1 + clean_dset2 + clean_dset3
        loader = DataLoader(dset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4)
        return loader# [0, 1], 256x256