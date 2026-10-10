"""
Датасет для задания. Цветной MNIST с ложной корреляцией.

Каждая цифра окрашена в один из 10 цветов. В обучающих данных цвет почти всегда
определяется классом (по умолчанию в 97% случаев цифра своего цвета):
сеть может научиться угадывать ответ по цвету вместо формы.
Проверка на данных, где цвет не связан с цифрой, показывает, чему она научилась на самом деле.

Выходная структура:

    dataset_colored/
        train/class_0/img_0_c0.png ...      обучающая выборка (смещённая)
        val/class_0/...                     валидация: то же распределение, что и train
        test_biased/class_0/...             тест с тем же смещением цвета
        test_unbiased/class_0/...           те же тестовые картинки, цвет не зависит от класса
        format.json                         {"class_0": 0, ..., "class_9": 9}
        meta.json                           палитра, параметры генерации, доля «своего» цвета в каждой выборке

В имени файла `img_{n}_c{k}.png` число k — номер цвета из палитры (meta.json),
n — номер картинки внутри класса. Картинки с одинаковыми n в test_biased и test_unbiased
(в одной папке класса) — одна и та же цифра в разных цветах.

Источник цифр — один из двух вариантов:
  1) папка в формате семинара (dataset/train/class_i/*.png, dataset/test/class_i/*.png):
         python prepare_colored_mnist.py --src dataset
  2) MNIST через torchvision (скачается автоматически):
         python prepare_colored_mnist.py

Дополнительно:
    --bias 0.97          доля цифр «своего» цвета в train/val/test_biased
    --max-per-class N    взять не более N картинок на класс (быстрая отладка)
    --seed 0             воспроизводимость
    --force              пересоздать существующую папку
"""
import argparse
import json
import os
import re
import shutil

import numpy as np
from PIL import Image

NUM_CLASSES = 10

PALETTE = [
    ("red", (255, 0, 0)),
    ("green", (0, 200, 0)),
    ("blue", (60, 60, 255)),
    ("yellow", (255, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("orange", (255, 128, 0)),
    ("purple", (150, 60, 255)),
    ("white", (255, 255, 255)),
    ("brown", (180, 100, 40)),
]


def parse_args():
    p = argparse.ArgumentParser(description="Цветной MNIST с ложной корреляцией цвета и класса")
    p.add_argument("--src", default=None,
                   help="папка с цифрами в формате семинара (train/, test/ с подпапками class_i); "
                        "без этого ключа MNIST скачивается через torchvision")
    p.add_argument("--raw", default="raw_data", help="папка для скачанного MNIST (если нет --src)")
    p.add_argument("--out", default="dataset_colored", help="папка для результата")
    p.add_argument("--bias", type=float, default=0.97,
                   help="доля цифр «своего» цвета в смещённых выборках (по умолчанию 0.97)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="доля исходного train, уходящая в val (по умолчанию 0.1)")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="не более N картинок на класс в каждой исходной выборке")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="удалить существующую папку --out")
    return p.parse_args()


def _natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def load_from_folder(root, max_per_class=None):
    """Читает root/class_i/*.png -> (список массивов uint8 28x28 по классам)."""
    per_class = []
    for i in range(NUM_CLASSES):
        cls_dir = os.path.join(root, f"class_{i}")
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Нет папки {cls_dir}")
        names = sorted((f for f in os.listdir(cls_dir) if f.lower().endswith(".png")), key=_natural_key)
        if max_per_class is not None:
            names = names[:max_per_class]
        imgs = []
        for name in names:
            img = Image.open(os.path.join(cls_dir, name)).convert("L")
            if img.size != (28, 28):
                raise ValueError(f"{name}: ожидается 28x28, получено {img.size}")
            imgs.append(np.asarray(img, dtype=np.uint8))
        per_class.append(imgs)
    return per_class


def load_from_torchvision(raw, train, max_per_class=None):
    import torchvision  # импорт здесь, чтобы режим --src работал без torchvision

    ds = torchvision.datasets.MNIST(raw, train=train, download=True)
    data = ds.data.numpy()
    targets = ds.targets.numpy()
    per_class = []
    for i in range(NUM_CLASSES):
        imgs = [np.asarray(x, dtype=np.uint8) for x in data[targets == i]]
        if max_per_class is not None:
            imgs = imgs[:max_per_class]
        per_class.append(imgs)
    return per_class


def colorize(gray, color_idx):
    """gray: uint8 HxW -> uint8 HxWx3, чёрный фон, штрих цвета PALETTE[color_idx]."""
    rgb = (gray[..., None].astype(np.float32) / 255.0) * np.array(PALETTE[color_idx][1], np.float32)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def sample_biased_color(label, rng, bias):
    if rng.random() < bias:
        return label
    others = [c for c in range(NUM_CLASSES) if c != label]
    return int(rng.choice(others))


def sample_uniform_color(rng):
    return int(rng.integers(NUM_CLASSES))


def write_split(split_dir, per_class, color_fn):
    """Сохраняет картинки, возвращает (число картинок по классам, число «своих» цветов)."""
    counts, own = [], 0
    for label, imgs in enumerate(per_class):
        out = os.path.join(split_dir, f"class_{label}")
        os.makedirs(out, exist_ok=True)
        for n, gray in enumerate(imgs):
            c = color_fn(label, n)
            Image.fromarray(colorize(gray, c)).save(os.path.join(out, f"img_{n}_c{c}.png"), "png")
            own += int(c == label)
        counts.append(len(imgs))
    return counts, own


def main():
    args = parse_args()
    if not 0.0 <= args.bias <= 1.0:
        raise SystemExit("--bias должен быть в диапазоне [0, 1]")

    if os.path.exists(args.out):
        if not args.force:
            print(f"Папка '{args.out}' уже существует. Запустите с --force, чтобы пересоздать.")
            return
        shutil.rmtree(args.out)

    if args.src:
        train_src = load_from_folder(os.path.join(args.src, "train"), args.max_per_class)
        test_src = load_from_folder(os.path.join(args.src, "test"), args.max_per_class)
    else:
        train_src = load_from_torchvision(args.raw, True, args.max_per_class)
        test_src = load_from_torchvision(args.raw, False, args.max_per_class)

    rng = np.random.default_rng(args.seed)

    # train / val: часть картинок каждого класса откладывается в val
    train_cls, val_cls = [], []
    for imgs in train_src:
        order = rng.permutation(len(imgs))
        n_val = int(round(len(imgs) * args.val_frac))
        val_cls.append([imgs[j] for j in order[:n_val]])
        train_cls.append([imgs[j] for j in order[n_val:]])

    # test_biased и test_unbiased строятся из одних и тех же картинок в одном порядке
    def biased(label, n):
        return sample_biased_color(label, rng, args.bias)

    def uniform(label, n):
        return sample_uniform_color(rng)

    os.makedirs(args.out)
    summary = {}
    for name, per_class, color_fn in (
        ("train", train_cls, biased),
        ("val", val_cls, biased),
        ("test_biased", test_src, biased),
        ("test_unbiased", test_src, uniform),
    ):
        counts, own = write_split(os.path.join(args.out, name), per_class, color_fn)
        total = sum(counts)
        summary[name] = {
            "images": total,
            "per_class": counts,
            "own_color_fraction": round(own / total, 4) if total else None,
        }

    with open(os.path.join(args.out, "format.json"), "w", encoding="utf-8") as fp:
        json.dump({f"class_{i}": i for i in range(NUM_CLASSES)}, fp)

    meta = {
        "palette": [{"index": i, "name": n, "rgb": list(rgb)} for i, (n, rgb) in enumerate(PALETTE)],
        "bias": args.bias,
        "seed": args.seed,
        "filename_pattern": "img_{n}_c{color_index}.png",
        "own_color_rule": "у класса i «свой» цвет имеет индекс i",
        "splits": summary,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print(f"Готово: {args.out}")
    for name, s in summary.items():
        print(f"  {name:14s} {s['images']:6d} картинок, доля «своего» цвета: {s['own_color_fraction']}")


if __name__ == "__main__":
    main()
