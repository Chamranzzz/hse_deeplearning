import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "primety.txt")


class PrimetyDataset(data.Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __getitem__(self, item):
        return self.lines[item]

    def __len__(self):
        return len(self.lines)


# ---------------------------------------------------------------------------
# Часть 1. Слой LoRA
# ---------------------------------------------------------------------------
#   h = W x + (alpha / r) * B A x
#   W: веса исходного слоя, заморожены
#   A: матрица (r, d_in),  маленькие случайные числа
#   B: матрица (d_out, r), нули

class LoRALayer(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base                     # исходный слой, его веса не обучаются

        if isinstance(base, nn.Linear):
            d_in, d_out = base.in_features, base.out_features
        else:
            # в GPT-2 вместо nn.Linear стоит слой Conv1D, он хранит веса как (d_in, d_out)
            d_in, d_out = base.weight.shape

        # создаем обучаемые матрицы A и B (nn.Parameter) и множитель scale.
        #   A: (r, d_in), случайные числа со стандартным отклонением 0.01
        #   B: (d_out, r), нули
        #   scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x):
        # x: (batch, T, d_in)
        # выход исходного слоя плюс поправка LoRA.
        # Поправка: x проходит через A, потом через B и умножается на scale.
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

# ---------------------------------------------------------------------------
# Часть 2. Замораживаем модель и вставляем адаптеры
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def sample(prompt, n=4):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt")
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        print("  ", tokenizer.decode(out[0], skip_special_tokens=True).replace("\n", " "))


print("--- до обучения ---")
sample("Примета: если")

total, trainable = count_params(model)
print(f"\nпараметров всего: {total}, обучаемых: {trainable}")

#   1) заморозим все параметры модели (requires_grad = False)
#   2) в каждом блоке трансформера заменим слой block.attn.c_attn на LoRALayer.
#      Блоки лежат в model.transformer.h, слой c_attn считает запросы, ключи и значения (Q, K, V)
for p in model.parameters():
    p.requires_grad = False

for block in model.transformer.h:
    block.attn.c_attn = LoRALayer(block.attn.c_attn, r=8, alpha=16)

total, trainable = count_params(model)
print(f"после вставки LoRA: всего {total}, обучаемых {trainable} ({100 * trainable / total:.2f}%)")

# ---------------------------------------------------------------------------
# Часть 3. Обучение
# ---------------------------------------------------------------------------

def collate(batch):
    # в конец каждой приметы добавляем токен конца текста, чтобы модель училась останавливаться
    enc = tokenizer([text + tokenizer.eos_token for text in batch], return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


d_train = PrimetyDataset(DATA_PATH)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=True, collate_fn=collate)
print("примеров в обучающей выборке:", len(d_train))
print("пример:", d_train[0])

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

epochs = 3
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for ids, mask in train_tqdm:
        logits = model(input_ids=ids, attention_mask=mask).logits      # (batch, T, vocab)

        # функция потерь языковой модели, сдвиг на один токен.
        #   прогнозы: логиты всех позиций, кроме последней
        #   ответы:   токены, начиная со второго
        #   в ответах на месте паддинга (mask == 0) поставим -100,
        #   F.cross_entropy с ignore_index=-100 такие позиции пропустит
        predict = logits[:, :-1, :]
        target = ids[:, 1:].clone()
        target[mask[:, 1:] == 0] = -100
        loss = F.cross_entropy(predict.reshape(-1, predict.size(-1)), target.reshape(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

# Вопрос: мы обучаем меньше четверти процента параметров, но один шаг идет почти столько же,
# сколько при полном дообучении. Почему? На чем тогда экономит LoRA?


# ---------------------------------------------------------------------------
# Часть 4. Что получилось
# ---------------------------------------------------------------------------

print("\n--- после обучения ---")
sample("Примета: если")
sample("Примета: если студент")

# сохраняем только адаптер: матрицы A и B из всех блоков
lora_state = {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
torch.save(lora_state, "lora_primety.tar")
print(f"\nадаптер весит {os.path.getsize('lora_primety.tar') / 1024 ** 2:.1f} МБ, вся модель около 500 МБ")

# Эксперименты:
#   попробуйте r = 1 и r = 64. Как меняются число параметров и качество?
#   попросите модель продолжить обычный текст ("Однажды утром"). Забыла ли она русский язык?
#   допишите в data/primety.txt десяток своих примет и переобучите


# ---------------------------------------------------------------------------
# Часть 5. Слияние адаптера с весами
# ---------------------------------------------------------------------------
# После обучения поправку можно прибавить к W один раз: W' = W + (alpha / r) * B A.
# Тогда модель снова состоит из обычных слоев и работает с прежней скоростью.

def merge(layer):
    """Вшивает адаптер в веса исходного слоя и возвращает исходный слой."""
    # посчитаем поправку delta = B A * scale, размер (d_out, d_in),
    # прибавим ее к весам layer.base. Не забудьте, что у Conv1D веса хранятся
    # как (d_in, d_out), то есть поправку нужно транспонировать.
    delta = (layer.B @ layer.A) * layer.scale
    with torch.no_grad():
        if isinstance(layer.base, nn.Linear):
            layer.base.weight += delta
        else:
            layer.base.weight += delta.T
    return layer.base


model.eval()
ids = tokenizer("Примета: если loss не падает", return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits_before = model(ids).logits

for block in model.transformer.h:
    block.attn.c_attn = merge(block.attn.c_attn)

with torch.no_grad():
    logits_after = model(ids).logits

total, trainable = count_params(model)
print("\nпараметров после слияния:", total)
print("максимальное расхождение логитов:", (logits_before - logits_after).abs().max().item())
# расхождение должно быть около нуля (порядка 1e-4): это одна и та же модель


# ---------------------------------------------------------------------------
# Бонус. То же самое через библиотеку peft
# ---------------------------------------------------------------------------
# pip install peft
#
# загрузим чистую модель заново и вставьте адаптеры через peft.
# Число обучаемых параметров должно совпасть с нашим: 294 912.
# Параметр fan_in_fan_out=True нужен из-за слоя Conv1D в GPT-2.
from peft import LoraConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    fan_in_fan_out=True,
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(base_model, config)
peft_model.print_trainable_parameters()

# Дальше peft_model обучается тем же циклом, что и выше.
# peft_model.save_pretrained("папка") сохранит только адаптер,
# peft_model.merge_and_unload() вернет обычную модель со вшитыми весами.
