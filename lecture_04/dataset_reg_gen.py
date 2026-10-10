import os
from random import randint
import json
import pygame

train_data = {'total': 10000, 'dir': "train"}
test_data = {'total': 1000, 'dir': "test"}
total_bk = 10
total_cls = 4
dir_out = 'dataset_reg'
file_format = 'format.json'
cls = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

if not os.path.exists(dir_out):
    os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

sun = pygame.image.load("images/sun64.png")
backs = [pygame.image.load(f"images/back_{n}.png") for n in range(1, total_bk+1)]

for info in (train_data, test_data):
    sun_coords = dict()

    for i in range(1, info['total']+1):
        file_out = f"sun_reg_{i}.png"
        im = backs[randint(0, total_bk-1)].copy()

        for _ in range(randint(20, 100)):
            x0 = randint(0, 256)
            y0 = randint(0, 256)
            pygame.draw.circle(im, cls[randint(0, total_cls-1)], (x0, y0), 1)

        x = randint(32, 256 - 32)
        y = randint(32, 256 - 32)
        sun_coords[file_out] = (x, y)
        im.blit(sun, (x-32, y-32))

        pygame.image.save(im, os.path.join(dir_out, info['dir'], file_out))

    fp = open(os.path.join(dir_out, info['dir'], file_format), "w")
    json.dump(sun_coords, fp)
    fp.close()
