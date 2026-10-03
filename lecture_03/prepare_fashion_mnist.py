import os
import json
import torchvision
import torchvision.transforms.v2 as tfs

# Fashion-MNIST: картинки одежды и обуви, 28x28, оттенки серого, 10 классов.
# Индексы классов:
#   0 - T-shirt/top (футболка)    5 - Sandal (сандалия)
#   1 - Trouser (брюки)           6 - Shirt (рубашка)
#   2 - Pullover (свитер)         7 - Sneaker (кроссовок)
#   3 - Dress (платье)            8 - Bag (сумка)
#   4 - Coat (пальто)             9 - Ankle boot (ботинок)

# сюда torchvision скачает исходные файлы датасета
root = r'C:\datasets\fashion_mnist'

transform = tfs.ToPILImage()

fashion_train = torchvision.datasets.FashionMNIST(root, download=True, train=True)
fashion_test = torchvision.datasets.FashionMNIST(root, download=True, train=False)

# отдельная папка, чтобы не перемешаться с датасетом MNIST, если он лежит рядом
dir_out = 'dataset_fashion'
file_format = 'format.json'
train_data = {'dir': "train", 'data': fashion_train}
test_data = {'dir': "test", 'data': fashion_test}

for info in (train_data, test_data):
    for i in range(10):
        os.makedirs(os.path.join(dir_out, info['dir'], f"class_{i}"), exist_ok=True)

for info in (train_data, test_data):
    for i in range(10):
        path = os.path.join(dir_out, info['dir'], f"class_{i}")
        cls = info['data'].data[info['data'].targets == i]

        for n, x in enumerate(cls):
            x = transform(x)
            x.save(os.path.join(path, f"img_{n}.png"), "png")

        print(f"{info['dir']}/class_{i}: {len(cls)} изображений")

targets = dict()
for i in range(10):
    targets[f'class_{i}'] = i

with open(os.path.join(dir_out, file_format), "w") as fp:
    json.dump(targets, fp)
