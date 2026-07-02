"""
Preprocessing pipeline untuk Gojek Sentiment Analysis.
Menggabungkan semua langkah cleaning dari notebook 01_eda.ipynb
menjadi module yang bisa dipakai ulang oleh API (FastAPI/Flask).

Cara pakai:
    from src.preprocessing import clean_text

    hasil = clean_text("Aplikasinya bagusss bgt, gk pernah error!")
"""

import re
import json
import os

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ============================================================
# 1. SLANG DICTIONARY
# (disalin persis dari notebook 01_eda.ipynb, Cell 19)
# ============================================================
SLANG_DICT = {
    # Negasi
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "g": "tidak",
    "tdk": "tidak", "nggak": "tidak", "ngga": "tidak", "ngg": "tidak",
    "kagak": "tidak", "kaga": "tidak", "engga": "tidak", "enggak": "tidak",
    "gx": "tidak", "kgk": "tidak",

    # Kata ganti
    "gw": "saya", "gue": "saya", "gua": "saya", "sy": "saya", "aku": "saya",
    "ku": "saya", "w": "saya",
    "lu": "kamu", "lo": "kamu", "loe": "kamu", "elu": "kamu",

    # Konjungsi & preposisi
    "yg": "yang", "yng": "yang",
    "dgn": "dengan", "dg": "dengan", "dng": "dengan",
    "tp": "tapi", "tpi": "tapi",
    "krn": "karena", "krna": "karena", "karna": "karena",
    "utk": "untuk", "buat": "untuk", "bt": "untuk",
    "dr": "dari", "dri": "dari",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau",
    "sm": "sama", "sma": "sama",
    "jd": "jadi", "jdi": "jadi",
    "jg": "juga", "jga": "juga",
    "jgn": "jangan", "jng": "jangan",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "dah": "sudah",
    "blm": "belum", "blum": "belum", "belom": "belum",

    # Kata sifat / keterangan
    "bgt": "banget", "bngt": "banget", "bngtt": "banget", "bgtt": "banget",
    "bgs": "bagus", "mantap": "bagus", "mantab": "bagus", "mantep": "bagus",
    "mantul": "bagus",
    "jelek": "buruk", "ancur": "hancur", "parah": "buruk",
    "bkn": "bukan",
    "emg": "memang", "emang": "memang", "mmg": "memang",
    "gmn": "bagaimana", "gimana": "bagaimana",
    "gtu": "begitu", "gitu": "begitu", "gt": "begitu",
    "gni": "begini", "gini": "begini",
    "lg": "lagi", "lgi": "lagi",
    "bs": "bisa", "bsa": "bisa",
    "aja": "saja", "aj": "saja", "doang": "saja", "doank": "saja",
    "byk": "banyak", "bnyk": "banyak",
    "skrg": "sekarang", "skrng": "sekarang", "skarang": "sekarang",
    "dl": "dulu", "dlu": "dulu",
    "bnr": "benar", "bner": "benar",
    "sllu": "selalu", "slalu": "selalu",
    "smpe": "sampai", "sampe": "sampai", "smpai": "sampai",
    "pdhl": "padahal",
    "cb": "coba",
    "ngmng": "ngomong", "ngmg": "ngomong",
    "brg": "bareng",
    "trs": "terus", "trus": "terus",
    "mksh": "terima kasih", "makasih": "terima kasih", "makasi": "terima kasih",
    "thx": "terima kasih", "thanks": "terima kasih", "thank": "terima kasih",
    "trims": "terima kasih",
    "tlg": "tolong",
    "pls": "tolong", "please": "tolong",

    # Kata kerja
    "nyoba": "coba", "cobain": "coba",
    "pake": "pakai", "pk": "pakai", "makenya": "pakai",
    "lemot": "lambat",
    "ngebug": "bug",

    # Domain aplikasi
    "apk": "aplikasi", "app": "aplikasi",
    "hp": "handphone", "hape": "handphone",
    "wa": "whatsapp",
    "rek": "rekening",

    # Lain-lain
    "org": "orang", "orng": "orang",
    "knp": "kenapa", "knpa": "kenapa",
    "dmn": "dimana", "dmna": "dimana",
    "kpn": "kapan",
    "smua": "semua", "smw": "semua",
    "hrs": "harus",
    "msh": "masih", "msi": "masih",
    "tgl": "tanggal",
    "nmr": "nomor", "nomer": "nomor",
    "pny": "punya",
    "lbh": "lebih",
    "drpd": "daripada",
    "bgmn": "bagaimana",
    "blg": "bilang",
    "sbnrnya": "sebenarnya", "sbnernya": "sebenarnya",
    "sbg": "sebagai",
}


# ============================================================
# 2. STOPWORDS (Sastrawi + custom, negasi dipertahankan)
# (disalin persis dari notebook 01_eda.ipynb, Cell 20)
# ============================================================
_stopword_factory = StopWordRemoverFactory()
STOPWORDS = set(_stopword_factory.get_stop_words())

NEGATION_WORDS = {"tidak", "bukan", "belum", "tanpa", "jangan", "nggak", "engga", "tak"}
STOPWORDS = STOPWORDS - NEGATION_WORDS

CUSTOM_STOPWORDS = {
    # Brand Gojek
    "gojek", "gopay", "goride", "gocar", "gofood", "gosend",
    "gopaylater", "goplay", "goclub", "gotix",

    # Partikel & filler
    "bintang", "dong", "deh", "sih", "nih", "loh", "ya", "yaa",
    "woi", "woy", "halo", "hai", "hi", "nya", "kan", "kok", "lho",

    # Kata umum non-sentimen
    "min", "admin", "biar", "banget", "cuma", "terus",
    "pakai", "mau", "makin", "lebih", "sekali", "sangat",
    "jadi", "kasih", "kali", "lama", "baru", "baik",
    "good", "nice", "the", "and",
}

STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)


# ============================================================
# 3. STEM CACHE (word_map)
# Di notebook, word_map dibangun dari kata unik dataset training
# (stopword -> "", kata lain -> hasil stem Sastrawi).
# Untuk API, kita load dari file cache yang sudah disimpan.
# ============================================================
_STEM_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'processed', 'stem_cache.json'
)

_stemmer = StemmerFactory().create_stemmer()

if os.path.exists(_STEM_CACHE_PATH):
    with open(_STEM_CACHE_PATH, 'r', encoding='utf-8') as f:
        WORD_MAP = json.load(f)
else:
    WORD_MAP = {}


def _stem_word(word: str) -> str:
    """
    Stem satu kata. Cek cache dulu (cepat), kalau tidak ada
    baru panggil Sastrawi langsung (lebih lambat, tapi ini
    yang menangani kata baru dari user yang belum pernah
    muncul di data training).
    """
    if word in WORD_MAP:
        return WORD_MAP[word]
    return _stemmer.stem(word)


# ============================================================
# 4. FUNGSI-FUNGSI PREPROCESSING
# (disalin persis dari notebook 01_eda.ipynb, Cell 22)
# ============================================================
def basic_clean(text: str) -> str:
    """Lowercase, hapus karakter non-alfabet, normalisasi spasi."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_repeated_chars(text: str) -> str:
    """'baguuuuus' -> 'bagus', 'kereeeen' -> 'keren'"""
    return re.sub(r'(.)\1{2,}', r'\1', text)


def normalize_slang(text: str) -> str:
    """Ganti kata slang/gaul ke bentuk formal."""
    words = text.split()
    return " ".join([SLANG_DICT.get(w, w) for w in words])


def apply_word_map(text: str) -> str:
    """
    Terapkan stopword removal + stemming.
    - Kata di STOPWORDS atau panjang < 3 huruf -> dihapus
    - Kata lain -> di-stem (via cache, atau langsung Sastrawi
      kalau kata itu belum ada di cache)
    """
    words = text.split()
    result = []
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        result.append(_stem_word(w))
    return " ".join(result)


# ============================================================
# 5. PIPELINE UTAMA
# Menggabungkan seluruh langkah di atas jadi satu fungsi
# ============================================================
def clean_text(text: str) -> str:
    """
    Pipeline preprocessing lengkap, urutan PERSIS sama dengan
    notebook 01_eda.ipynb:
        1. basic_clean            - lowercase, hapus simbol
        2. normalize_repeated_chars - "baguuus" -> "bagus"
        3. normalize_slang        - "gk" -> "tidak"
        4. apply_word_map         - stopword removal + stemming

    Dipakai baik saat training (lewat notebook) maupun saat
    inference di API, supaya hasilnya konsisten.
    """
    text = basic_clean(text)
    text = normalize_repeated_chars(text)
    text = normalize_slang(text)
    text = apply_word_map(text)
    return text