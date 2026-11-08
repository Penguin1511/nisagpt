"""Türkçe Vikipedi'den küçük bir eğitim metni hazırlar."""
from pathlib import Path

from datasets import load_dataset

HEDEF_KARAKTER = 5_000_000          # yaklaşık 5 milyon karakter
EN_KISA_MAKALE = 500                # bundan kısa sayfaları atla
CIKTI = Path("data/tr_wiki.txt")


def main():
    CIKTI.parent.mkdir(exist_ok=True)   # data/ klasörü yoksa oluştur

    # streaming=True: tüm Vikipedi'yi indirmez, makaleleri tek tek getirir
    ds = load_dataset("wikimedia/wikipedia", "20231101.tr",
                      split="train", streaming=True)

    toplam = 0
    makale = 0
    with CIKTI.open("w", encoding="utf-8") as f:
        for ornek in ds:
            metin = ornek["text"].strip()
            if len(metin) < EN_KISA_MAKALE:
                continue
            f.write(metin + "\n\n")
            toplam += len(metin)
            makale += 1
            if toplam >= HEDEF_KARAKTER:
                break

    print(f"{makale} makale, {toplam:,} karakter yazıldı -> {CIKTI}")


if __name__ == "__main__":
    main()