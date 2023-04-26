"""Helper classes and functions used among the code."""

import time
import datetime
import random
import os
import unittest
from io import StringIO
from unittest.mock import patch
from typing import Dict, Callable, Any, List

import numpy as np


__all__ = ["Timer", "ProgressBar", "TestHelper", "gen_seeds", "split_list",
           "split_range", "print_banner_line", "print_banner_block",
           "get_datetime"]


class Timer:
    """
    Class for recording the time usage of function calls within programs.

    Attributes
    ----------
    total_time: Dict[str, float]
        overall time usage
    start_time: Dict[str, float]
        time of last self.tic() call
    end_time: Dict[str, float]
        time of last self.toc() call
    """
    def __init__(self) -> None:
        self.total_time = {}
        self.start_time = {}
        self.end_time = {}

    def tic(self, slot: str) -> None:
        """
        Begin tracking time usage and store it in a slot.

        :param slot: name of the slot
        :return: None
        """
        if slot not in self.total_time.keys():
            self.total_time[slot] = 0.0
        self.start_time[slot] = time.time()

    def toc(self, slot: str) -> None:
        """
        Stop tracking time usage store it in a slot.

        :param slot: name of the slot
        :return: None
        """
        if slot not in self.start_time.keys():
            raise RuntimeError("Record for slot '%s' not started!" % slot)
        else:
            self.end_time[slot] = time.time()
            self.total_time[slot] += self.end_time[slot] - \
                self.start_time[slot]

    def report_time(self) -> None:
        """
        Report time usage between two self.tic() and self.toc() calls and reset
        self.start_time and self.end_time.

        :return: None
        """
        max_slot_length = max([len(slot) for slot in self.start_time.keys()])
        indent = "%3s" % ""
        time_format = "%%%ds : %%10.2fs" % max_slot_length
        for slot in self.start_time.keys():
            if slot in self.end_time.keys():
                print(indent, time_format %
                      (slot, self.end_time[slot]-self.start_time[slot]))
            else:
                raise RuntimeError("Record for slot '%s' not ended!" % slot)
        self.reset()

    def report_total_time(self) -> None:
        """
        Report overall time usage.

        :return: None
        """
        max_slot_length = max([len(slot) for slot in self.total_time.keys()])
        indent = "%3s" % ""
        time_format = "%%%ds : %%10.2fs" % max_slot_length
        for slot in self.total_time.keys():
            print(indent, time_format % (slot, self.total_time[slot]))
        self.reset_total()

    def reset(self) -> None:
        """
        Reset self.start_time and self.end_time for next measurement.

        :return: None
        """
        self.start_time = {}
        self.end_time = {}

    def reset_total(self) -> None:
        """
        Same as reset(), also resets self.total_time.

        :return: None
        """
        self.start_time = {}
        self.end_time = {}
        self.total_time = {}


class ProgressBar:
    """
    Class for reporting the progress for a time-consuming task.

    Attributes
    ----------
    num_count: int
        total amount of tasks
        For example, if you are going to do some calculation for a vector with
        length of 1000, then set it to 1000.
    num_scales: int
        total number of scales in the progress bar
        For example, if you want to split the whole task into 10 parts, then set
        it to 10. When one part finishes, the program will report 10% of the
        whole task has been finished.
    scale_unit: float
        amount of tasks between two ad-joint scales
        See the schematic plot below for demonstration.
    next_scale: float
        the next scale waiting for counter
        If counter exceeds next_scale, then we can know one part of the task
        finish. See the schematic plot for demonstration.
    counter: int
        counter in range of [1, num_count]
    num_scales_past: int
        number of past scales
    percent_unit: float
        percentage of tasks between two ad-joint scales

    NOTES
    -----
    Schematic plot of the working mechanism:
        num_count:  50
        num_scales: 10
        scale_unit: 5
        scales:     0----|----|----|----|----|----|----|----|----|----|
        next_scale: ....................* -> * -> * -> * -> * -> * -> *
        counter:    ................^
        num_scales_past: 3
        percent_unit: 10%
    """
    def __init__(self, num_count: int, num_scales: int = 10) -> None:
        """
        :param num_count: total amount of tasks
        :param num_scales: total number of scales in the progress bar
        """
        self.num_count = num_count
        self.num_scales = num_scales
        self.scale_unit = num_count / num_scales
        self.next_scale = self.scale_unit
        self.counter = 0
        self.num_scales_past = 0
        self.percent_unit = 100 / num_scales

    def count(self) -> None:
        """
        Increase the counter by 1.

        :return: None
        """
        self.counter += 1
        if self.counter >= self.next_scale:
            self.num_scales_past += 1
            self.next_scale += self.scale_unit
            print("[%3d%%] finished %6d of %6d" %
                  (self.num_scales_past * self.percent_unit,
                   self.counter, self.num_count))


class TestHelper:
    """
    Helper class that makes unittest easier.

    Attributes
    ----------
    tester: unittest.TestCase
        TestCase instance upon which tests are performed
    """
    def __init__(self, tester: unittest.TestCase) -> None:
        """
        :param tester: instance of unittest.TestCase
        """
        self.tester = tester

    def test_equal_array(self, array1: np.ndarray,
                         array2: np.ndarray,
                         almost: bool = False,
                         delta: float = 1.0e-7) -> None:
        """
        Checks if two arrays are equal to each other.

        :param array1: 1st array to compare
        :param array2: 2nd array to compare
        :param almost: whether to compare according to difference
        :param delta: threshold for difference
        :return: None.
        """
        diff = np.sum(np.abs(array1 - array2)).item(0)
        if not almost:
            self.tester.assertEqual(diff, 0)
        else:
            self.tester.assertAlmostEqual(diff, 0.0, delta=delta)

    def test_no_equal_array(self, array1: np.ndarray,
                            array2: np.ndarray,
                            almost: bool = False,
                            delta: float = 1.0e-7) -> None:
        """
        Checks if two arrays are not equal to each other.

        :param array1: 1st array to compare
        :param array2: 2nd array to compare
        :param almost: whether to compare according to difference
        :param delta: threshold for difference
        :return: None.
        """
        diff = np.sum(np.abs(array1 - array2)).item(0)
        if not almost:
            self.tester.assertNotEqual(diff, 0)
        else:
            self.tester.assertNotAlmostEqual(diff, 0.0, delta=delta)

    def test_raise(self, func: Callable, exception: Any, message: str) -> None:
        """
        Tests if expected exception is raised during an operation.

        :param func: wrapper function over the operation to test
        :param exception: category of exception to test
        :param message: expected exception message
        :return: None
        """
        with self.tester.assertRaises(exception) as cm:
            func()
        if message is not None:
            self.tester.assertRegex(str(cm.exception), message)

    def test_no_raise(self, func: Callable, exception: Any) -> None:
        """
        Tests if expected exception is not raised during an operation.

        :param func: wrapper function over the operation to test
        :param exception: category of exception to test
        :return: None
        """
        try:
            func()
        except exception:
            status = True
        else:
            status = False
        self.tester.assertFalse(status)

    def test_stdout(self, func: Callable, message: List[str]) -> None:
        """
        Test if the output contain given message.

        :param func: wrapper function over the operation to test
        :param message: reference message with which to compare, each regex
            corresponds to one line of output
        :return: None
        """
        with patch('sys.stdout', new=StringIO()) as fake_out:
            func()
        output = [out for out in fake_out.getvalue().split("\n") if out != ""]
        for i, msg in enumerate(message):
            self.tester.assertRegex(output[i], msg)


def gen_seeds(num_seeds: int) -> List[int]:
    """
    Generate a list of random numbers from /dev/urandom as seeds for Python
    random number generator.

    :param num_seeds: number of seeds
    :return seeds: generated seeds
    """
    num_bytes = [random.randint(1, 4) for _ in range(num_seeds)]
    byte_order = ['big' if random.randint(1, 2) == 1 else 'little'
                  for _ in range(num_seeds)]
    # signed = [True if random.randint(1, 2) == 1 else False
    #           for _ in range(num_seeds)]
    signed = [True for _ in range(num_seeds)]
    seeds = []
    for item in zip(num_bytes, byte_order, signed):
        rand_int = int().from_bytes(os.urandom(item[0]), byteorder=item[1],
                                    signed=item[2])
        if rand_int not in seeds:
            seeds.append(rand_int)
    return seeds


def split_list(raw_list: List[Any],
               num_group: int,
               algorithm: str = "remainder") -> List[List[Any]]:
    """
    Split given list into different groups.

    Two algorithms are implemented: by the remainder of the index of each
    element divided by the number of group, or the range of index. For example,
    if we are to split the list of [0, 1, 2, 3] into two groups, by remainder
    we will get [[0, 2], [1, 3]] while by range we will get [[0, 1], [2, 3]].

    :param raw_list: incoming list to split
    :param num_group: number of groups
    :param algorithm: algorithm for grouping elements, should be either
        "remainder" or "range"
    :return: split list from raw_list
    """
    assert num_group in range(1, len(raw_list)+1)
    assert algorithm in ("remainder", "range")
    num_element = len(raw_list)
    if algorithm == "remainder":
        list_split = [[raw_list[i] for i in range(num_element)
                      if i % num_group == k] for k in range(num_group)]
    else:
        # Get the numbers of items for each group
        num_item = [num_element // num_group for _ in range(num_group)]
        for i in range(num_element % num_group):
            num_item[i] += 1
        # Divide the list according to num_item
        list_split = []
        for i in range(num_group):
            j0 = sum(num_item[:i])
            j1 = j0 + num_item[i]
            list_split.append([raw_list[j] for j in range(j0, j1)])
    return list_split


def split_range(n_max: int, num_group: int = 1) -> List[range]:
    """
    Split range(n_max) into different groups.

    Adapted from split_list with algorithm = "range".

    :param n_max: upperbound of range, starting from 0
    :param num_group: number of groups
    :return: list of ranges split from range(n_max)
    """
    # Get the numbers of items for each group
    num_item = [n_max // num_group for _ in range(num_group)]
    for i in range(n_max % num_group):
        num_item[i] += 1
    range_list = []
    for i in range(num_group):
        j0 = sum(num_item[:i])
        j1 = j0 + num_item[i]
        range_list.append(range(j0, j1))
    return range_list


def print_banner_line(text: str,
                      width: int = 80,
                      mark: str = "-",
                      end: str = "#") -> None:
    """
    Print a banner like '#--------------- FOO ---------------#' to stdout.

    :param text: central text in the banner
    :param width: total width of the banner
    :param mark: border character of the banner
    :param end: end character prepended and appended to the banner
    :return: None
    """
    num_marks_total = width - len(text) - 4
    num_marks_left = num_marks_total // 2
    num_marks_right = num_marks_total - num_marks_left
    banner_with_marks = end + mark * num_marks_left
    banner_with_marks += " %s " % text
    banner_with_marks += mark * num_marks_right + end
    print(banner_with_marks)


def print_banner_block(text: str,
                       width: int = 80,
                       mark: str = "-",
                       end: str = "#") -> None:
    """
    Print a banner like
    #----------------------------------#
    #               FOO                #
    #----------------------------------#
    to stdout.

    :param text: central text in the banner
    :param width: total width of the banner
    :param mark: border character of the banner
    :param end: end character prepended and appended to the banner
    :return: None
    """
    num_spaces_total = width - len(text) - 2
    num_spaces_left = num_spaces_total // 2
    num_spaces_right = num_spaces_total - num_spaces_left
    banner_with_spaces = end + " " * num_spaces_left
    banner_with_spaces += text
    banner_with_spaces += " " * num_spaces_right + end
    border = end + mark * (width - 2) + end
    print(border)
    print(banner_with_spaces)
    print(border)


def get_datetime(fmt: str = "%x %X") -> str:
    """
    Return current date and time.

    :param fmt: date and time format
    :return: current date and time
    """
    now = datetime.datetime.now()
    return now.strftime(fmt)
