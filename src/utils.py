import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
from architecture import AllConvNet
class NormalizeNegativeImages(object):

    def __call__(self, item):
        min_value_pixel = torch.min(item)
        if min_value_pixel < 0:
            item -= min_value_pixel
            item /= torch.max(item)
        return item
    
def Transform_image(dataset_path):
    # Semilla para estandarizar resultados
    torch.manual_seed(2320)

    cuda = cuda and torch.cuda.is_available()
    trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    train_mean = trainset.data.mean(axis=(0, 1, 2)) / 255
    train_std = trainset.data.std(axis=(0, 1, 2)) / 255

    # Data normal (32x32)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        NormalizeNegativeImages(),
        transforms.RandomCrop(32, padding=4, antialias=False),
        transforms.RandomHorizontalFlip(antialias=False)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        NormalizeNegativeImages()
    ])

    #  transformaciones a 8x8 y de vuelta a 32x32 por bilineal

    transform_test8x8 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        NormalizeNegativeImages(),
        transforms.Resize((8,8)),
        transforms.Resize((32,32)),
    ])

    transform_train8x8 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        NormalizeNegativeImages(),
        transforms.Resize((8,8)),
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])

    # Crearemos transformaciones sin randomizar nada para la resta de imágenes.

    transform_train_no_random = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        NormalizeNegativeImages()
    ])

    transform_train8x8_no_random = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        NormalizeNegativeImages(),
        transforms.Resize((8,8)),
        transforms.Resize((32,32))
    ])

    return transform_train, transform_test, transform_train8x8, transform_test8x8, transform_train_no_random, transform_train8x8_no_random



def load_data(dataset_path, train_batch_size, test_batch_size, transform_train, transform_test):

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # Data normal

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root=dataset_path, train=True, download=True,
        transform=transform_train),
        batch_size=train_batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=dataset_path, train=False, download=True,
        transform=transform_test),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    

    return train_loader, test_loader




def train(epoch: int, loader: torch.utils.data.DataLoader,
          model: AllConvNet, optimizer):

    global criterion
    global cuda

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 150 == 0:
          print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(loader.dataset),
              100. * batch_idx / len(loader), loss.item()))


def test(epoch: int, best_loss: int, best_epoch: int,
         test_loader: torch.utils.data.DataLoader, model: AllConvNet,
         test_name=''):

    global criterion
    global cuda

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        acc = 100 * correct / len(test_loader.dataset)

    test_loss /= len(test_loader.dataset)

    print(
        '\n{} Test set: Average loss: {:.4f}, Accuracy: {} ({:.0f}%)\n'.format(
            test_name, test_loss, acc, 100. * correct /
            len(test_loader.dataset)))


    if test_loss < best_loss:
        best_epoch = epoch
        best_loss = test_loss

    return best_loss, best_epoch, acc