from tokenizer import UNK, CharTokenizer

ORNEK = "merhaba dünya, ağaç ışık İstanbul göl"


def test_geri_donus():
    tok = CharTokenizer(ORNEK, en_az=1)
    assert tok.decode(tok.encode(ORNEK)) == ORNEK


def test_turkce_karakterler_sozlukte():
    tok = CharTokenizer(ORNEK, en_az=1)
    for ch in "ğüşıöçİ":
        assert ch in tok.stoi


def test_sozluk_sirali():
    tok = CharTokenizer(ORNEK, en_az=1)
    karakterler = [tok.itos[i] for i in range(tok.vocab_size)]
    assert karakterler == sorted(karakterler)


def test_bilinmeyen_karakter_unk_olur():
    tok = CharTokenizer("merhaba", en_az=1)
    assert tok.decode(tok.encode("maz")) == "ma" + UNK


def test_nadir_karakter_unk_olur():
    tok = CharTokenizer("aaab", en_az=2)        # 'b' sadece 1 kez geçiyor
    assert "b" not in tok.stoi
    assert tok.decode(tok.encode("ab")) == "a" + UNK


def test_sozluk_boyutu_ve_numaralar():
    tok = CharTokenizer("aab", en_az=1)
    assert tok.vocab_size == 3                          # a, b, UNK
    tok = CharTokenizer("aa" + UNK, en_az=1)            # metinde zaten UNK var
    assert tok.vocab_size == 2                          # a, UNK
    assert sorted(tok.stoi.values()) == [0, 1]          # numaralar 0'dan boşluksuz