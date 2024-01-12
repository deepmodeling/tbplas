#! /usr/bin/env python
import re
import glob
import sys


def conv32_to_64(src):
    patterns = [re.compile(r"^\s*(INTEGER)\s*::", re.I),
                re.compile(r"^\s*(INTEGER)\s*,", re.I)]

    with open(src, "r") as src_file:
        content = src_file.readlines()

    for i, line in enumerate(content):
        for pattern in patterns:
            result = re.search(pattern, line)
            if result is not None:
                if result.group(1)[:3] == "INT":
                    content[i] = re.sub(r"INTEGER", "INTEGER(KIND=8)", line)
                else:
                    content[i] = re.sub(r"integer", "integer(kind=8)", line)

    with open(src, "w") as src_file:
        src_file.writelines(content)


def conv64_to_32(src):
    patterns = [re.compile(r"^\s*(INTEGER\(KIND=8\))\s*::", re.I),
                re.compile(r"^\s*(INTEGER\(KIND=8\))\s*,", re.I)]

    with open(src, "r") as src_file:
        content = src_file.readlines()

    for i, line in enumerate(content):
        for pattern in patterns:
            result = re.search(pattern, line)
            if result is not None:
                if result.group(1)[:3] == "INT":
                    content[i] = re.sub(r"INTEGER\(KIND=8\)", "INTEGER", line)
                else:
                    content[i] = re.sub(r"integer\(kind=8\)", "integer", line)

    with open(src, "w") as src_file:
        src_file.writelines(content)


def main():
    try:
        target = int(sys.argv[1])
    except IndexError:
        target = 64
    if target == 32:
        func = conv64_to_32
    elif target == 64:
        func = conv32_to_64
    else:
        print(f"Illegal integer type {sys.argv[1]}")
        sys.exit(-1)

    for postfix in ("F90", "f90"):
        src_files = glob.glob(f"*.{postfix}")
        for src in src_files:
            func(src)


if __name__ == "__main__":
    main()
