import argparse
import random
import sys
import math

# ==========================================
# GROUP LOGIC IMPLEMENTATIONS
# ==========================================

class GroupStrategy:
    def generate_element(self):
        raise NotImplementedError
    def operate(self, a, b, op_symbol):
        raise NotImplementedError
    def format_element(self, el):
        return str(el)
    def max_unique(self):
        return float('inf')

# --- 1. Integers (Z) ---
class IntegerGroup(GroupStrategy):
    def __init__(self, min_val, max_val, allow_mult):
        self.min = min_val
        self.max = max_val
        self.ops = ['+']
        if allow_mult: self.ops.append('*')

    def generate_element(self):
        return random.randint(self.min, self.max)

    def operate(self, a, b, op_symbol):
        if op_symbol == '+': return a + b
        if op_symbol == '*': return a * b
        return 0

    def get_op(self):
        return random.choice(self.ops)

    def max_unique(self):
        range_size = self.max - self.min + 1
        return (range_size ** 2) * len(self.ops)

# --- 2. Cyclic Group (Z_n) ---
class CyclicGroup(GroupStrategy):
    def __init__(self, modulus):
        self.mod = modulus

    def generate_element(self):
        return random.randint(0, self.mod - 1)

    def operate(self, a, b, op_symbol):
        # Cyclic group usually implies addition modulo N
        return (a + b) % self.mod

    def get_op(self):
        return '+'

    def max_unique(self):
        return self.mod ** 2

# --- 3. Symmetric Group (S_n) ---
class PermutationGroup(GroupStrategy):
    def __init__(self, n):
        self.n = n
        # Base list 1..n
        self.elements = list(range(1, n + 1))

    def generate_element(self):
        # Return a tuple for hashability in 'seen' set
        el = self.elements[:]
        random.shuffle(el)
        return tuple(el)

    def operate(self, a, b, op_symbol):
        # Composition: (a * b)[i] = a[b[i]]
        # Note: We use 1-based values, but 0-based indexing
        result = []
        for i in range(self.n):
            # b[i] gives the value at index i. 
            # We use that value (minus 1) as the index for a.
            inner_val = b[i]
            res_val = a[inner_val - 1]
            result.append(res_val)
        return tuple(result)

    def format_element(self, el):
        # If single digits, squash them: "123"
        if self.n <= 9:
            return "".join(map(str, el))
        # If double digits, space them: "10 11 12"
        return " ".join(map(str, el))

    def get_op(self):
        return '*'

    def max_unique(self):
        # Order of S_n is n!
        # Max lines = (n!)^2
        factorial = math.factorial(self.n)
        return factorial ** 2

# --- 4. Dihedral Group (D_2n) ---
class DihedralGroup(GroupStrategy):
    def __init__(self, n):
        self.n = n

    def generate_element(self):
        # Element is (rotation, reflection)
        # r in [0, n-1], s in [0, 1]
        r = random.randint(0, self.n - 1)
        s = random.randint(0, 1)
        return (r, s)

    def operate(self, a, b, op_symbol):
        r1, s1 = a
        r2, s2 = b
        
        # Logic: r^a s^b * r^c s^d
        # relation: s r = r^-1 s  => s r^k = r^-k s
        
        if s1 == 0:
            # No flip on left: rotations add normally
            return ((r1 + r2) % self.n, s2)
        else:
            # Flip on left: r1 s * r2 s2
            # = r1 (s r2) s2 = r1 (r^-2 s) s2 = r^(r1-r2) s^(1+s2)
            return ((r1 - r2) % self.n, (1 + s2) % 2)

    def format_element(self, el):
        r, s = el
        # Format: r2s1 or r0s0
        # Optimized: if r0, don't show r. if s0, don't show s. e for identity.
        # BUT: For consistency in training, standard dense notation is often better.
        # Let's use explicit "rX sY" notation for clarity.
        # Or compact: "r2s" (flip) "r2" (no flip)
        s_str = "s" if s == 1 else ""
        return f"r{r}{s_str}"

    def get_op(self):
        return '*'

    def max_unique(self):
        # Order is 2n. Max lines = (2n)^2
        order = 2 * self.n
        return order ** 2

# ==========================================
# MAIN SCRIPT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate sophisticated algebraic datasets.")
    
    # Selection
    parser.add_argument('--group', type=str, required=True, 
                        choices=['integers', 'cyclic', 'permutation', 'dihedral'],
                        help="The algebraic structure to sample from.")
    
    # Configuration
    parser.add_argument('--count', type=int, required=True, help='Number of unique lines.')
    parser.add_argument('--n', type=int, default=10, 
                        help="Context parameter N (Modulus for Cyclic, Size for Perm/Dihedral).")
    parser.add_argument('--out', type=str, default='input.txt', help='Output filename.')
    
    # Specifics
    parser.add_argument('--mult', type=str, default='false', help="Include multiplication (only for 'integers').")
    parser.add_argument('--min', type=int, default=0, help="Min int (only for 'integers').")
    parser.add_argument('--max', type=int, default=100, help="Max int (only for 'integers').")

    args = parser.parse_args()

    # Initialize Strategy
    strategy = None
    if args.group == 'integers':
        allow_mult = args.mult.lower() == 'true'
        strategy = IntegerGroup(args.min, args.max, allow_mult)
        print(f"Group: Integers (Range {args.min}-{args.max})")
    elif args.group == 'cyclic':
        strategy = CyclicGroup(args.n)
        print(f"Group: Cyclic (Mod {args.n})")
    elif args.group == 'permutation':
        strategy = PermutationGroup(args.n)
        print(f"Group: Symmetric S_{args.n}")
    elif args.group == 'dihedral':
        strategy = DihedralGroup(args.n)
        print(f"Group: Dihedral D_{2*args.n} (Symmetries of {args.n}-gon)")

    # Safety Check
    max_lines = strategy.max_unique()
    if args.count > max_lines:
        print(f"Error: Requested {args.count} lines, but group only supports {max_lines} unique combinations.")
        sys.exit(1)

    # Generation Loop
    seen = set()
    print(f"Generating {args.count} unique samples...")
    
    try:
        with open(args.out, 'w') as f:
            while len(seen) < args.count:
                a = strategy.generate_element()
                b = strategy.generate_element()
                op = strategy.get_op()
                
                # Tuple for uniqueness check
                # We must store the ELEMENTS in the seen set, not the string representation
                # to ensure mathematical uniqueness.
                triplet = (a, b, op)
                
                if triplet not in seen:
                    seen.add(triplet)
                    
                    c = strategy.operate(a, b, op)
                    
                    # Formatting
                    a_str = strategy.format_element(a)
                    b_str = strategy.format_element(b)
                    c_str = strategy.format_element(c)
                    
                    f.write(f"{a_str}{op}{b_str}={c_str}\n")
                    
                    if len(seen) % 5000 == 0:
                        print(f"Generated {len(seen)} lines...", end='\r')

        print(f"\nDone. Saved to {args.out}")

    except IOError as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
