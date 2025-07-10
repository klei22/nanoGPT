import pytest
try:
    from yakinori import Yakinori
    yakinori = Yakinori()
except Exception:
    pytest.skip("yakinori not available", allow_module_level=True)
sentence = "幽遊白書は最高の漫画です"
parsed_list = yakinori.get_parsed_list(sentence)
# print(parsed_list)
hiragana_sentence = yakinori.get_hiragana_sentence(parsed_list)
print(hiragana_sentence)
