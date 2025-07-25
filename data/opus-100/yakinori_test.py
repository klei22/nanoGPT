import pytest

try:
    from yakinori import Yakinori
    yakinori = Yakinori()
except Exception as e:
    pytest.skip(f"yakinori unavailable: {e}", allow_module_level=True)


def test_yakinori_basic():
    sentence = "幽遊白書は最高の漫画です"
    parsed_list = yakinori.get_parsed_list(sentence)
    hiragana_sentence = yakinori.get_hiragana_sentence(parsed_list)
    assert isinstance(hiragana_sentence, str)
