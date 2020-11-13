from torchvision.datasets import ImageFolder


class KaggleCastingDataset(ImageFolder):

    def __init__(self, *args, **kwargs):
        super(KaggleCastingDataset, self).__init__(*args, **kwargs)
