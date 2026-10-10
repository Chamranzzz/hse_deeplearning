"""
Датасет для задания 2

10 классов (самолёт, машина, птица, кот, олень, собака, лягушка, лошадь, корабль, грузовик),
цветные картинки 3x32x32, 50 000 в train и 10 000 в test.

Формат такой же, как у датасета с семинара:

    dataset_cifar/
        train/class_0/img_0.png ...
        test/class_0/...
        format.json          {"class_0": 0, ..., "class_9": 9}
        class_names.json     {"0": "airplane", ...}

Картинки цветные: открывайте их через Image.open(path).convert("RGB"), в сети in_channels=3.

Скрытый тест для турнира «бюджет параметров» (часть D задания 2)
----------------------------------------------------------------
Ключ --hidden-out DIR делит тест пополам внутри каждого класса:
  - чётные картинки остаются в <out>/test (публичный тест, 5000 картинок) — его раздаёте студентам;
  - нечётные уходят в <hidden-out>/test (скрытый тест, 5000 картинок) — его храните у себя
    и прогоняете через evaluate_submission.py.
Без этого ключа весь тест (10 000) остаётся в <out>/test.

Запуск:
    python prepare_cifar10.py
    python prepare_cifar10.py --hidden-out dataset_cifar_hidden
    python prepare_cifar10.py --per-class-train 1000 --force      # урезанный train для отладки
"""
import argparse
import json
import os
import shutil

import torchvision

NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    p = argparse.ArgumentParser(description="Экспорт CIFAR-10 в папки с PNG")
    p.add_argument("--raw", default="raw_data",
                   help="куда torchvision скачает исходные файлы (по умолчанию raw_data)")
    p.add_argument("--out", default="dataset_cifar",
                   help="папка для готового датасета (по умолчанию dataset_cifar)")
    p.add_argument("--hidden-out", default=None,
                   help="папка для скрытой половины теста (для турнира); без ключа тест не делится")
    p.add_argument("--per-class-train", type=int, default=None,
                   help="оставить в train не более N картинок на класс (для быстрых экспериментов)")
    p.add_argument("--force", action="store_true",
                   help="удалить существующие папки --out и --hidden-out и создать заново")
    return p.parse_args()


def make_class_dirs(split_dir):
    for i in range(NUM_CLASSES):
        os.makedirs(os.path.join(split_dir, f"class_{i}"), exist_ok=True)


def save(img, split_dir, label, n):
    img.save(os.path.join(split_dir, f"class_{label}", f"img_{n}.png"), "png")


def export_split(ds, split_dir, limit=None, hidden_dir=None):
    """Сохраняет картинки в split_dir/class_i/img_n.png.

    Если задан hidden_dir, нечётные по порядку картинки каждого класса уходят туда.
    Возвращает (число картинок по классам в split_dir, то же для hidden_dir).
    """
    make_class_dirs(split_dir)
    if hidden_dir:
        make_class_dirs(hidden_dir)

    seen = [0] * NUM_CLASSES        # сколько картинок класса прочитано
    kept = [0] * NUM_CLASSES        # сколько записано в split_dir
    hidden = [0] * NUM_CLASSES      # сколько записано в hidden_dir

    for img, label in ds:  # img - PIL.Image (RGB), label - int
        k = seen[label]
        seen[label] += 1
        if hidden_dir and k % 2 == 1:
            save(img, hidden_dir, label, hidden[label])
            hidden[label] += 1
            continue
        if limit is not None and kept[label] >= limit:
            continue
        save(img, split_dir, label, kept[label])
        kept[label] += 1
    return kept, hidden


def write_meta(root):
    with open(os.path.join(root, "format.json"), "w", encoding="utf-8") as fp:
        json.dump({f"class_{i}": i for i in range(NUM_CLASSES)}, fp)
    with open(os.path.join(root, "class_names.json"), "w", encoding="utf-8") as fp:
        json.dump({str(i): name for i, name in enumerate(CLASS_NAMES)}, fp, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    targets = [args.out] + ([args.hidden_out] if args.hidden_out else [])
    existing = [t for t in targets if os.path.exists(t)]
    if existing:
        if not args.force:
            print(f"Уже существует: {', '.join(existing)}. Запустите с --force, чтобы пересоздать.")
            return
        for t in existing:
            shutil.rmtree(t)
    for t in targets:
        os.makedirs(t)

    train = torchvision.datasets.CIFAR10(args.raw, train=True, download=True)
    test = torchvision.datasets.CIFAR10(args.raw, train=False, download=True)

    train_counts, _ = export_split(train, os.path.join(args.out, "train"), args.per_class_train)
    hidden_test = os.path.join(args.hidden_out, "test") if args.hidden_out else None
    test_counts, hidden_counts = export_split(test, os.path.join(args.out, "test"), hidden_dir=hidden_test)

    for t in targets:
        write_meta(t)

    print(f"Готово: {args.out}")
    print(f"  train: {sum(train_counts)} картинок, по классам: {train_counts}")
    print(f"  test:  {sum(test_counts)} картинок, по классам: {test_counts}")
    if args.hidden_out:
        print(f"Скрытый тест: {args.hidden_out}/test, {sum(hidden_counts)} картинок "
              f"(храните у себя, студентам не раздавайте)")


if __name__ == "__main__":
    main()
