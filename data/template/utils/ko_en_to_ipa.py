import subprocess
from konlpy.tag import Okt
import argparse
import re

def transcribe_korean(sentence):
    """Transcribe a Korean sentence into its phonemes using KoNLPy (Okt)."""
    okt = Okt()
    tokens = okt.morphs(sentence)
    tokenized_sentence = ' '.join(tokens)

    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "ko", "--ipa", tokenized_sentence],
            capture_output=True,
            text=True
        )

        # Remove unwanted characters
        transcription = result.stdout.strip().replace("ㆍ"," ")
        # Check for failed transcription markers
        if "(en)" in transcription or "(ko)" in transcription:
            return "[[[[[" + sentence + "]]]]]"# Return original sentence on failure
        return transcription
    except Exception as e:
        return f"Error in transcribing Korean: {str(e)}"

def handle_mixed_language(word):
    """Handle a word with potential Korean, Language, or number content."""
    if word.isdigit():  # Detect numbers but just pass through for now (different in each language)
        return word
    elif any('가' <= char <= '힣' for char in word):  # Detect Korean
        return transcribe_korean(word)
    else:  # Non-Korean Word
        return "[[[[[" + word + "]]]]]"

def transcribe_multilingual(sentences, output_file):
    """Transcribe multilingual sentences (English and Korean, with numbers) and save to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            result = []
            # Split sentence but keep punctuation (preserve spaces, commas, etc.)
            words = re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)
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
    parser.add_argument('input_file', type=str, help='Path to the input file containing sentences.')
    parser.add_argument('output_file', type=str, help='Path to the output file for IPA transcription.')

    args = parser.parse_args()

    # Read input sentences
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    # Transcribe and save to the output file
    transcribe_multilingual(sentences, args.output_file)

if __name__ == '__main__':
    main()


