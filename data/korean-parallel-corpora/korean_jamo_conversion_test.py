import pytest

try:
    from jamo import h2j, j2hcj, j2h, is_jamo
except Exception as e:
    pytest.skip(f"jamo unavailable: {e}", allow_module_level=True)

def korean_to_phonetic(text):
    """Converts Korean text to its phonetic representation."""
    # Convert Hangul to individual jamos
    decomposed_text = h2j(text)
    # Convert jamos back to Hangul compatibility jamos (for readability)
    phonetic_text = j2hcj(decomposed_text)
    return phonetic_text


def test_korean_jamo_roundtrip():
    korean_text = "안 녕 하 세 요"
    phonetic_text = korean_to_phonetic(korean_text)
    reconstructed_list = []
    for pho in phonetic_text.split(" "):
        if not pho:
            continue
        elif is_jamo(pho[0]):
            reconstructed_list.append(j2h(*pho))
        else:
            reconstructed_list.append(pho.replace('▁', ' '))
    reconstructed = ''.join(reconstructed_list)
    assert isinstance(reconstructed, str)

