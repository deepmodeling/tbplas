#! /usr/bin/env python
import re
import glob


def convert_int(src):
    patterns = [re.compile(r"^[\t ]*INTEGER[ ]*::"),
                re.compile(r"^[\t ]*INTEGER[ ]*,"),
                re.compile(r"^[\t ]*integer[ ]*::"),
                re.compile(r"^[\t ]*integer[ ]*,")]

    with open(src, "r") as src_file:
        content = src_file.readlines()

    for i, line in enumerate(content):
        for j, pattern in enumerate(patterns):
            if re.search(pattern, line) is not None:
                if j in (0, 1):
                    content[i] = re.sub(r"INTEGER", "INTEGER(KIND=8)", line)
                else:
                    content[i] = re.sub(r"integer", "integer(kind=8)", line)

    with open(src, "w") as src_file:
        src_file.writelines(content)


def main():
    for postfix in ("F90", "f90"):
        src_files = glob.glob(f"*.{postfix}")
        for src in src_files:
            convert_int(src)


if __name__ == "__main__":
    main()
