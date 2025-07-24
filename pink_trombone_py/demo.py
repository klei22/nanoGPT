import argparse
from .ipa_mapping import synthesize_to_wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo pink trombone synthesis')
    parser.add_argument('text', help='Input text')
    parser.add_argument('--out', default='out.wav', help='Output wav file')
    args = parser.parse_args()
    synthesize_to_wav(args.text, args.out)
    print(f'Wrote {args.out}')
