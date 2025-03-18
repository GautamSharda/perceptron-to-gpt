import sys

def unidimensional_linear_function(m, b, x):
    return m*x + b

if __name__ == "__main__":
    # take in -k as an argument 
    if len(sys.argv) != 3:
        print("Usage: python data_generator.py <input_range_start> <input_range_end>")
        sys.exit(1)
    xs = int(sys.argv[1])
    xe = int(sys.argv[2])
    data = [unidimensional_linear_function(1, 0, x) for x in range(xs, xe)]
    # write input and output to csv
    with open("data.csv", "w") as f:
        for x, y in zip(range(xs, xe), data):
            f.write(f"{x},{y}\n")
