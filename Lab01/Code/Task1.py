def my_or(x1, x2):
    return x1 or x2

def my_and(x1, x2):
    return x1 and x2

def my_xor(x1, x2):
    return my_or(x1, x2) and not my_and(x1, x2)

def main():
    numbers = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]

    print("Результати XOR для 5 пар чисел:")
    for i in range(0, 10, 2):
        x1_num, x2_num = numbers[i], numbers[i + 1]
        x1, x2 = bool(numbers[i]), bool(numbers[i + 1])
        result = my_xor(x1, x2)
        print(f"XOR({x1_num}, {x2_num}) = {result}")

if __name__ == "__main__":
    main()
