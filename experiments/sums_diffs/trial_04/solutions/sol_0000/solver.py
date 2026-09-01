"""Official SimpleTES 17-element initial construction."""

import json

INITIAL_SET = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]


def main():
    with open("solution.json", "w") as stream:
        json.dump({"A": INITIAL_SET}, stream)
    print(f"wrote official SimpleTES seed: n={len(INITIAL_SET)}")


if __name__ == "__main__":
    main()
