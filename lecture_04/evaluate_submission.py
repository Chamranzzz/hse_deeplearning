"""
У вас папка с двумя файлами:

    submission/
        model.py       содержит функцию build_model() -> torch.nn.Module
        weights.pt     state_dict, сохранённый через torch.save(model.state_dict(), "weights.pt")

Контракт модели:
  - вход: float32-тензор [B, 3, 32, 32] со значениями в [0, 1] (без нормализации);
    нормализация (если нужна) выполняется внутри модели;
  - выход: логиты [B, 10].

"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

NUM_CLASSES = 10


def parse_args():
    p = argparse.ArgumentParser(description="Оценка работы в турнире «бюджет параметров»")
    p.add_argument("--submission", required=True, help="папка с model.py и weights.pt")
    p.add_argument("--data", required=True, help="папка теста с подпапками class_0 ... class_9")
    p.add_argument("--max-params", type=int, default=150_000, help="бюджет параметров (по умолчанию 150000)")
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def load_images(root):
    """Читает root/class_i/*.png -> (тензор [N,3,32,32] float32 в [0,1], метки [N])."""
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        d = os.path.join(root, f"class_{c}")
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Нет папки {d}")
        for name in sorted((f for f in os.listdir(d) if f.lower().endswith(".png")), key=natural_key):
            img = np.asarray(Image.open(os.path.join(d, name)).convert("RGB"), dtype=np.uint8)
            if img.shape != (32, 32, 3):
                raise ValueError(f"{name}: ожидается 32x32 RGB, получено {img.shape}")
            xs.append(img)
            ys.append(c)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    return x, torch.tensor(ys, dtype=torch.long)


def load_model(submission_dir):
    model_path = os.path.join(submission_dir, "model.py")
    weights_path = os.path.join(submission_dir, "weights.pt")
    for path in (model_path, weights_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Не найден файл {path}")

    spec = importlib.util.spec_from_file_location("student_model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.abspath(submission_dir))  # чтобы работали импорты рядом с model.py
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise AttributeError("В model.py нет функции build_model()")

    model = module.build_model()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def accuracy(model, x, y, batch, device):
    model.eval().to(device)
    correct = 0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch].to(device))
        if logits.shape[-1] != NUM_CLASSES:
            raise ValueError(f"Ожидается {NUM_CLASSES} логитов, получено {tuple(logits.shape)}")
        correct += (logits.argmax(1).cpu() == y[i:i + batch]).sum().item()
    return correct / len(x)


def main():
    args = parse_args()
    model = load_model(args.submission)
    params = sum(p.numel() for p in model.parameters())

    x, y = load_images(args.data)
    t0 = time.time()
    acc = accuracy(model, x, y, args.batch, args.device)
    elapsed = time.time() - t0

    result = {
        "submission": os.path.basename(os.path.abspath(args.submission)),
        "params": params,
        "max_params": args.max_params,
        "within_budget": params <= args.max_params,
        "images": len(x),
        "accuracy": round(acc, 4),
        "eval_seconds": round(elapsed, 2),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
