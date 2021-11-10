from data.base_dataset import BaseDataset

class FFHQDataset(BaseDataset):
    def __init__(self, opt, is_inference):
        super(FFHQDataset, self).__init__(opt, is_inference)


    def __len__(self):
        return len(self.image_index)

    def get_image_index(self, is_inference):
        if is_inference:
            return list(range(10000))
        else:
            return list(range(10000, 70000))
