import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import basic_clean, normalize_repeated_chars, normalize_slang


def test_basic_clean_lowercase():
    assert basic_clean("BAGUS sekali!!!") == "bagus sekali"


def test_basic_clean_removes_symbols():
    assert basic_clean("aplikasi error 404???") == "aplikasi error"


def test_normalize_repeated_chars():
    assert normalize_repeated_chars("baguuuus") == "bagus"
    assert normalize_repeated_chars("kereeeen") == "keren"


def test_normalize_slang():
    assert normalize_slang("gk bs login") == "tidak bisa login"