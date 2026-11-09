"""Karakter seviyesinde tokenizer."""
from collections import Counter

UNK = "\ufffd"   # "�": Unicode'un kendi "bilinmeyen karakter" işareti


class CharTokenizer:
    def __init__(self, metin, en_az=10):
        """Metinde en az `en_az` kez geçen karakterlerden bir sözlük kurar.
        Daha nadir karakterler ve sözlükte olmayan her şey UNK olur."""
        # 1. Her karakterin kaç kez geçtiğini say.
        sayac = Counter(metin)
        # 2. En az `en_az` kez geçen karakterleri bir kümeye (set) al.
        kume = {ch for ch, say in sayac.items() if say >= en_az}
        # 3. Kümeye UNK'yi ekle.
        kume.add(UNK)
        # 4. Kümeyi sırala ve self.karakterler'e ata.
        self.karakterler = sorted(kume)
        # 5. self.stoi ve self.itos sözlüklerini kur.
        self.stoi = {ch: i for i, ch in enumerate(self.karakterler)}
        self.itos = {i: ch for i, ch in enumerate(self.karakterler)}
        # 6. self.unk_id = UNK'nin numarası
        self.unk_id = self.stoi[UNK]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, metin):
        """Metni sayı listesine çevirir. Sözlükte olmayan karakter UNK olur."""
        return [self.stoi.get(ch, self.unk_id) for ch in metin]

    def decode(self, ids):
        """Sayı listesini metne çevirir."""
        return "".join(self.itos[i] for i in ids)