from src.utils import safe_num, convert_damage_to_float

def test_safe_num_valid():
    assert safe_num("123") == 123.0

def test_safe_num_invalid():
    assert safe_num("abc") == 0.0

def test_damage_millions():
    assert convert_damage_to_float("$1.5M") == 1_500_000.0

def test_damage_billions():
    assert convert_damage_to_float("$2B") == 2_000_000_000.0

def test_damage_empty():
    assert convert_damage_to_float("") == 0.0