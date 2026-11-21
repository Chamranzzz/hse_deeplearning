import math

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

print("параметров в модели:", sum(p.numel() for p in model.parameters()))
print("размер словаря:", model.config.vocab_size)


# ---------------------------------------------------------------------------
# Часть 1. Токены
# ---------------------------------------------------------------------------

text = "Нейросети учатся предсказывать следующее слово"
ids = tokenizer(text, return_tensors="pt")["input_ids"]     # тензор (1, T)

print("\nтекст:", text)
print("размер тензора:", tuple(ids.shape))
print("номера токенов:", ids[0].tolist())
print("токены:", [tokenizer.decode([i]) for i in ids[0].tolist()])

# ---------------------------------------------------------------------------
# Часть 2. Распределение следующего токена
# ---------------------------------------------------------------------------

def next_token_probs(text):
    """Вероятности следующего токена после текста: тензор размера (vocab,)."""
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits            # (1, T, vocab): прогноз для КАЖДОЙ позиции
    # Делаем прогноз только после последнего токена
    last = logits[0, -1]
    probs = F.softmax(last, dim=-1)
    return probs


def show_top(text, k=8):
    probs = next_token_probs(text)
    top = torch.topk(probs, k)
    print(f"\n{text} ...")
    for p, i in zip(top.values, top.indices):
        print(f"   {p.item():.3f}  {tokenizer.decode([i.item()])!r}")


show_top("Утром я выпил чашку")
show_top("Столица России - город")
show_top("Дважды два равно")
show_top("На лекции по нейросетям преподаватель рассказывал про")


# ---------------------------------------------------------------------------
# Часть 3. Свой цикл генерации
# ---------------------------------------------------------------------------

def generate(text, max_new_tokens=30, temperature=1.0, top_k=None):
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids).logits[0, -1]         # логиты следующего токена, (vocab,)

        if temperature == 0:
            # Жадный выбор
            next_id = torch.argmax(logits).view(1)
        else:
            # Сэмплирование.
            #   1) разделяем логиты на температуру
            #   2) если задан top_k, оставляем только top_k самых больших логитов,
            #      остальным поставьте -inf torch.topk
            #   3) softmax и один бросок кубика: torch.multinomial(probs, num_samples=1)
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # авторегрессия. Дописываем новый токен в конец ids и идем на следующий круг.
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0])


start = "Однажды студент пришел на экзамен и"

print("\n--- жадно (temperature = 0) ---")
print(generate(start, temperature=0))

print("\n--- temperature = 0.7, top_k = 50 ---")
for _ in range(3):
    print(generate(start, temperature=0.7, top_k=50))

print("\n--- temperature = 1.5, без top_k ---")
print(generate(start, temperature=1.5))


# ---------------------------------------------------------------------------
# Часть 4. Детектор испорченных пословиц
# ---------------------------------------------------------------------------
# Контекст задачки :
# Языковая модель умеет оценивать, насколько текст "ожидаемый".
# Удивление модели от токена: -log p(токен | все предыдущие).
# Среднее удивление по тексту под экспонентой называется перплексией.
# Это та же самая функция потерь, на которой модель обучалась.

def token_surprisal(text):
    """Список пар (токен, удивление) для всех токенов, начиная со второго."""
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits[0]              # (T, vocab)
    # сдвиг на один токен
    #   прогноз на позиции t сравнивается с настоящим токеном на позиции t + 1
    #   1) log_softmax от логитов всех позиций, кроме последней
    #   2) targets: все токены, кроме первого
    #   3) для каждой позиции достаем лог-вероятность правильного токена и поменяем знак
    log_probs = F.log_softmax(logits[:-1], dim=-1)          # (T-1, vocab)
    targets = ids[0, 1:]                                    # (T-1,)
    nll = -log_probs[torch.arange(len(targets)), targets]   # (T-1,)
    return [(tokenizer.decode([t.item()]), s.item()) for t, s in zip(targets, nll)]


def perplexity(text):
    # Перплексия = exp(среднее удивление по токенам)
    values = [s for _, s in token_surprisal(text)]
    return math.exp(sum(values) / len(values))


PROVERBS = [
    ("Без труда не выловишь и рыбку из пруда", "Без труда не выловишь и рыбку из шкафа"),
    ("Не имей сто рублей, а имей сто друзей", "Не имей сто рублей, а имей сто кастрюль"),
    ("Семь раз отмерь, один раз отрежь", "Семь раз отмерь, один раз покрась"),
    ("Тише едешь, дальше будешь", "Тише едешь, дальше уснешь"),
    ("Яблоко от яблони недалеко падает", "Яблоко от яблони недалеко летает"),
    ("Утро вечера мудренее", "Утро вечера зеленее"),
]

print("\n--- детектор испорченных пословиц ---")
caught = 0
for original, broken in PROVERBS:
    ppl_original = perplexity(original)
    ppl_broken = perplexity(broken)
    caught += ppl_broken > ppl_original

    # самый неожиданный токен в испорченной версии
    token, value = max(token_surprisal(broken), key=lambda pair: pair[1])
    print(f"{ppl_original:9.1f}  {original}")
    print(f"{ppl_broken:9.1f}  {broken}   <- модель споткнулась на {token!r} ({value:.1f})")

print(f"\nмодель распознала подделку в {caught} случаях из {len(PROVERBS)}")


# ---------------------------------------------------------------------------
# Часть 5. То же самое через библиотеку
# ---------------------------------------------------------------------------
# Все, что мы написали руками, спрятано в одном методе.

enc = tokenizer(start, return_tensors="pt")
out = model.generate(
    **enc,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)
print("\n--- model.generate ---")
print(tokenizer.decode(out[0], skip_special_tokens=True))

