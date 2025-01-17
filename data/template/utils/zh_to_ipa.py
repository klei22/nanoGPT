import subprocess
from dragonmapper import hanzi
import jieba
import argparse
import re
import json

def transcribe_chinese(sentence):
    """Transcribe a Chinese sentence into its phonemes using dragonmapper."""
    try:
        result = hanzi.to_ipa(sentence)
        return "".join(result)
    except Exception as e:
        return f"Error in transcribing Chinese: {str(e)}"

def handle_mixed_language(word):
    """Handle a word with potential Chinese, Language, or number content."""
    if word.isdigit():  # Detect numbers but just pass through for now (different in each language)
        return word
    elif any(hanzi.is_simplified(char) for char in word):  # Detect Chinese
        return transcribe_chinese(word)
    else:  # Non-Chinese Word
        return "[[[[[" + word + "]]]]]"

def transcribe_multilingual(lists, output_file):
    """Transcribe multilingual sentences (English and Chinese, with numbers) and save to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in lists:
            result = []
            sentence = item['sentence']
            # Split sentence using jieba
            seg_list = jieba.cut(sentence, cut_all=False)
            seg_sentence = " ".join(seg_list)
            # Split sentence but keep punctuation (preserve spaces, commas, etc.)
            words = re.findall(r'\w+|[^\w\s]', seg_sentence, re.UNICODE)
            for word in words:
                if re.match(r'\w+', word):  # Only process words (skip punctuation)
                    result.append(handle_mixed_language(word))
                else:
                    result.append(word)  # Preserve punctuation as is
            transcription_result = " ".join(result)
            f.write(transcription_result + "\n")
            print(transcription_result)  # Print to console for reference

def main():
    parser = argparse.ArgumentParser(description='Transcribe multilingual sentences into IPA phonemes.')
    parser.add_argument('input_file', type=str, help='Path to the input file containing sentences in json format.')
    parser.add_argument('output_file', type=str, help='Path to the output file for IPA transcription.')

    args = parser.parse_args()

    # Read input sentences
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lists = json.load(f)

    # Transcribe and save to the output file
    transcribe_multilingual(lists, args.output_file)

if __name__ == '__main__':
    main()
