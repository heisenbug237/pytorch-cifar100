
import os
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from torchvision import transforms as tfms
import torchvision.datasets as Datasets
from datasets import DatasetFactory
from utils import seed_all, show_batch, count_parameters, validate 
from models import resnet
import opendatasets as od
import argparse


models = ['resnet20', 'resnet32', 'resnet56']
weights_url_map = {
    'resnet20' : 'https://github.com/heisenbug237/pytorch-cifar100/releases/download/pretrained/resnet20_cifar100-23dac2f1.pt',
    'resnet32' : 'https://github.com/heisenbug237/pytorch-cifar100/releases/download/pretrained/resnet32_cifar100-84213ce6.pt',
    'resnet56' : 'https://github.com/heisenbug237/pytorch-cifar100/releases/download/pretrained/resnet56_cifar100-f2eff4c8.pt',
}
model_functor_map = {
    'resnet20' : resnet.make_resnet20,
    'resnet32' : resnet.make_resnet32,
    'resnet56' : resnet.make_resnet56,
}

# augmentations
stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
train_tfms = tfms.Compose([
    tfms.RandomCrop(32, padding=4, padding_mode='reflect'),
    tfms.RandomHorizontalFlip(),
    tfms.ToTensor(),
    tfms.Normalize(*stats, inplace=True)
])
val_tfms = tfms.Compose([
    tfms.ToTensor(),
    tfms.Normalize(*stats)
])
img_transforms = {
    'train': train_tfms, 
    'val': val_tfms, 
    'test': val_tfms
    }
target_transforms = {
    'train': None, 
    'val': None, 
    'test': None
    }

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='running parameters',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_source', default='trailmet', type=str, 
                help='source for cifar100 dataset', choices=['kaggle', 'trailmet', 'torchvision'])
    parser.add_argument('--batch_size', default=256, type=int, 
                help='mini-batch size for data loader')
    parser.add_argument('--seed', default=69, type=int, 
                help='random seed for results reproduction')
    parser.add_argument('--test_pretrained', action='store_true', default=False,
                help='test pretarined model')
    args=parser.parse_args()
    seed_all(args.seed)

    if args.data_source=='kaggle': 
        od.download('https://www.kaggle.com/minbavel/cifar-100-images', data_dir='../')
        data_dir = '../cifar-100-images/CIFAR100'
        classes = os.listdir(data_dir+'/TRAIN')
        print('Total Classes = ', len(classes))

        train_ds = Datasets.ImageFolder(data_dir+'/TRAIN', img_transforms['train'])
        test_ds = Datasets.ImageFolder(data_dir+'/TEST', img_transforms['val'])

        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_ds, args.batch_size, pin_memory=True)

        # show_batch(val_loader, 100, 1, stats)
        print('Dataset Loaded')
                                                             
    elif args.data_source=='torchvision':
        cifar100_train = Datasets.CIFAR100(root='./data', train=True, download=True, transform=train_tfms)
        cifar100_test = Datasets.CIFAR100(root='./data', train=False, download=True, transform=val_tfms)

        train_loader = DataLoader(cifar100_train, shuffle=True, num_workers=1, batch_size=args.batch_size)
        test_loader = DataLoader(cifar100_test, shuffle=False, num_workers=1, batch_size=args.batch_size)
        print('Dataset Loaded')
        
    elif args.data_source=='trailmet':
        cifar_dataset = DatasetFactory.create_dataset(
            name = 'CIFAR100', 
            root = os.getcwd(),
            split_types = ['train', 'val', 'test'],
            val_fraction = 0.2,
            transform = img_transforms,
            target_transform = target_transforms
            )
        train_loader = DataLoader(
                cifar_dataset['train_dataset'], batch_size=args.batch_size, 
                sampler=cifar_dataset['train_sampler'],
                num_workers=0
            )
        test_loader = DataLoader(
                cifar_dataset['test_dataset'], batch_size=args.batch_size, 
                sampler=cifar_dataset['test_sampler'],
                num_workers=0
            )
        print('Dataset Loaded')
        
    else:
        print('Invalid Data Source')

    if args.test_pretrained:
        for model in models:
            print('==> Using Model :', model)
            cnn = model_functor_map[model](100, 32)
            count_parameters(cnn, print_table=False)
            state_dict = load_state_dict_from_url(weights_url_map[model], progress=True)
            cnn.load_state_dict(state_dict)
            cnn.eval()
            acc1 = []
            for batch in test_loader:
                acc1.append(validate(cnn, batch))
            sums=0
            for acc in acc1:
                sums+=acc.item()
            print('Top-1 Accuracy :',sums*100/len(acc1))
            del cnn
