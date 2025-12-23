#!/bin/bash


python3 utils/en2ipa.py text_eng_Latn.txt --mode text --output_file ipa_text_eng_Latn.txt
python3 utils/ja2ipa.py text_jpn_Jpan.txt ipa_text_jpn_Jpan.txt --text_output --use_spacy --text_no_sentence
python3 utils/ko_en_to_ipa.py text_kor_Hang.txt --text_input --text_output ipa_text_kor_Hang.txt
python3 utils/zh_to_ipa.py text_zho_Hans.txt ipa_text_zho_Hans.txt --input_type text --no-wrapper
