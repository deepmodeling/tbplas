#! /usr/bin/env python
import re


def replace(filename, patterns, new_lines):
    with open(filename, "r") as in_file:
        content = in_file.readlines()
    for i, pattern in enumerate(patterns):
        pattern = re.compile(pattern)
        for j, line in enumerate(content):
            if re.search(pattern, line) is not None:
                content[j] = new_lines[i]
    with open(filename, "w") as out_file:
        out_file.writelines(content)


def main():
    new_version = input("Input the short version number: ")
    new_release = input("Input the full version number: ")

    # Update setup.py
    patterns = [r"^[\t ]*'version'"]
    new_lines = [f"{'':4}'version': '{new_release}',\n"]
    replace("setup.py", patterns, new_lines)

    # Update setup.py
    patterns = [r"^version =", r"^release ="]
    new_lines = [f"version = '{new_version}'\n", f"release = '{new_release}'\n"]
    replace("doc/source/conf.py", patterns, new_lines)


if __name__ == "__main__":
    main()
