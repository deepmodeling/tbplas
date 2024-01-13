#! /usr/bin/env python
"""Update version number, date, etc. in various files"""
import re
import datetime


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


def get_nl(content):
    line_numbers = [i for i, line in enumerate(content)
                    if re.search(r"^\s*\*", line) is not None]
    return min(line_numbers), max(line_numbers)


def update_readme():
    with open("README.rst", "r") as in_file:
        readme = in_file.readlines()
    with open("doc/source/about/index.rst", "r") as in_file:
        features = in_file.readlines()
    r0, r1 = get_nl(readme)
    f0, f1 = get_nl(features)
    del(readme[r0:r1+1])
    readme[r0-1] = "\n" + "".join(features[f0:f1+1])
    with open("README.rst", "w") as in_file:
        in_file.writelines(readme)


def main():
    new_version = input("Input the short version number: ")
    new_release = input("Input the full version number: ")

    # Update version in setup.py
    patterns = [r"^\s*'version'"]
    new_lines = [f"{'':4}'version': '{new_release}',\n"]
    replace("setup.py", patterns, new_lines)

    # Update version in pyproject.toml
    patterns = [r"^\s*version\s*="]
    new_lines = [f"version = \"{new_release}\"\n"]
    replace("pyproject.toml", patterns, new_lines)

    # Update version and date in doc/source/conf.py
    year = datetime.datetime.today().year
    patterns = [r"^version\s*=", r"^release\s*=", r"^copyright\s*="]
    new_lines = [f"version = '{new_version}'\n",
                 f"release = '{new_release}'\n",
                 f"copyright = '2017-{year}, TBPLaS development team. All rights reserved'\n"]
    replace("doc/source/conf.py", patterns, new_lines)

    # Update date in doc/source/about/license.rst
    patterns = [r"^Copyright"]
    new_lines = [f"Copyright (c) 2017-{year}, TBPLaS development team. All rights reserved.\n"]
    replace("doc/source/about/license.rst", patterns, new_lines)

    # Update README from documentation
    update_readme()


if __name__ == "__main__":
    main()
