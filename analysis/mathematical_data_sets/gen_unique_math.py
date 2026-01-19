import argparse
import random
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate unique integer arithmetic dataset (Addition & optional Multiplication).")
    
    # Core parameters
    parser.add_argument('--count', type=int, required=True, help='Number of unique lines to generate')
    parser.add_argument('--out', type=str, default='input.txt', help='Output filename')
    
    # Range params (Only used if modulus is 'infty')
    parser.add_argument('--min', type=int, default=0, help="Min value (used only if modulus='infty')")
    parser.add_argument('--max', type=int, default=100, help="Max value (used only if modulus='infty')")
    
    # Modulus argument
    parser.add_argument('--modulus', type=str, default='infty', 
                        help="Modulus N (integer) or 'infty'. If N is set, inputs are forced to [0, N-1].")
    
    # Multiplication toggle
    parser.add_argument('--mult', type=str, default='false',
                        help="Set to 'true' to include multiplication (*) in the dataset.")
    
    args = parser.parse_args()

    # 1. Setup Operations
    ops = ['+']
    if args.mult.lower() == 'true':
        ops.append('*')
        print(f"Operations enabled: Addition (+) and Multiplication (*)")
    else:
        print(f"Operations enabled: Addition (+) only")

    # 2. Determine Mode and Effective Ranges
    modulus = None
    effective_min = 0
    effective_max = 0

    if args.modulus.lower() in ['infty', 'inf', 'none', 'standard']:
        # --- INFINITE MODE ---
        modulus = None
        effective_min = args.min
        effective_max = args.max
        print(f"Mode: Standard Arithmetic (Range [{effective_min}, {effective_max}])")
    else:
        # --- FINITE MODULAR MODE ---
        try:
            modulus = int(args.modulus)
            if modulus <= 0:
                raise ValueError("Modulus must be positive.")
            
            effective_min = 0
            effective_max = modulus - 1
            
            print(f"Mode: Modular Arithmetic (Z_{modulus})")
            print(f"Constraint: Inputs forced to range [0, {effective_max}]")
            
            if args.min != 0 or args.max != 100:
                print("Notice: --min and --max arguments are ignored when Modulus is finite.")

        except ValueError as e:
            print(f"Error parsing modulus: {e}")
            sys.exit(1)

    # 3. Safety Check: Calculate total possible unique combinations
    range_size = effective_max - effective_min + 1
    # Total unique lines = (Range * Range) * (Number of Operators)
    max_possible = (range_size * range_size) * len(ops)

    if args.count > max_possible:
        print(f"Error: You requested {args.count} unique lines, but the available space "
              f"only allows for {max_possible} unique combinations.")
        sys.exit(1)

    print(f"Generating {args.count} unique samples...")
    
    # 4. Generation Loop
    seen = set()
    try:
        with open(args.out, 'w') as f:
            while len(seen) < args.count:
                a = random.randint(effective_min, effective_max)
                b = random.randint(effective_min, effective_max)
                op = random.choice(ops)
                
                # Uniqueness is now based on the triplet (a, b, op)
                # distinct from (a, b) alone because 2+2 != 2*2
                triplet = (a, b, op)
                
                if triplet not in seen:
                    seen.add(triplet)
                    
                    # Calculate Result
                    result = 0
                    if op == '+':
                        result = a + b
                    elif op == '*':
                        result = a * b
                    
                    # Apply modulus if needed
                    if modulus:
                        result = result % modulus
                    
                    f.write(f"{a}{op}{b}={result}\n")
                    
                    if len(seen) % 10000 == 0 and len(seen) > 0:
                        print(f"Generated {len(seen)} lines...", end='\r')

        print(f"\nDone. {len(seen)} unique lines saved to {args.out}")

    except IOError as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
