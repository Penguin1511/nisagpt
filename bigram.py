"""Faz 2: Bigram dil modeli."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import CharTokenizer

# --- Ayarlar (hiperparametreler) ---
batch_size = 32
block_size = 8
max_iters = 5000        # toplam eğitim adımı
eval_interval = 500     # kaç adımda bir loss'u ölçelim
eval_iters = 200        # loss'u ölçerken kaç batch'in ortalamasını alalım
learning_rate = 1e-2    # her adımda sayılar ne kadar değişsin (0.01)
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Veri ---
with open("data/tr_wiki.txt", encoding="utf-8") as f:
    metin = f.read()
tok = CharTokenizer(metin)
veri = torch.tensor(tok.encode(metin), dtype=torch.long)
n = int(0.9 * len(veri))
train_veri = veri[:n]
val_veri = veri[n:]


def get_batch(bolum):
    """Eğitim ya da doğrulama verisinden rastgele bir batch döndürür."""
    kaynak = train_veri if bolum == "train" else val_veri
    ix = torch.randint(len(kaynak) - block_size, (batch_size,))
    x = torch.stack([kaynak[i:i + block_size] for i in ix])
    y = torch.stack([kaynak[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tablo = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.tablo(idx)                        # (B, T, V)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, kac_karakter):
        """idx: (B, T) başlangıç metni. Her adımda sonuna bir karakter ekler."""
        for _ in range(kac_karakter):
            logits, _ = self(idx)                       # (B, T, V)
            logits = logits[:, -1, :]                   # sadece son konum: (B, V)
            olasiliklar = F.softmax(logits, dim=-1)     # skorları olasılığa çevir
            sonraki = torch.multinomial(olasiliklar, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, sonraki], dim=1)      # (B, T + 1)
        return idx


model = BigramModel(tok.vocab_size).to(device)

# Eğitimden önce model ne üretiyor?
baslangic = torch.tensor([[tok.stoi["\n"]]], device=device)
uretilen = model.generate(baslangic, kac_karakter=200)
print(tok.decode(uretilen[0].tolist()))

@torch.no_grad()
def estimate_loss():
    """Train ve val kümelerinde ortalama loss'u ölçer."""
    sonuc = {}
    for bolum in ["train", "val"]:
        kayiplar = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(bolum)
            _, loss = model(X, Y)
            kayiplar[k] = loss.item()
        sonuc[bolum] = kayiplar.mean().item()
    return sonuc


# --- Eğitim ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for adim in range(max_iters):
    if adim % eval_interval == 0:
        kayip = estimate_loss()
        print(f"adım {adim}: train loss {kayip['train']:.4f}, val loss {kayip['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

kayip = estimate_loss()
print(f"son durum: train loss {kayip['train']:.4f}, val loss {kayip['val']:.4f}")

# Eğitimden sonra model ne üretiyor?
print("--- Eğitimden sonra ---")
uretilen = model.generate(baslangic, kac_karakter=500)
print(tok.decode(uretilen[0].tolist()))

