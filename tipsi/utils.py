"""utils.py contains some helper classes and functions.

Classes
-------
    Timer:
        Tracking time usage of function calls.
    ProgressBar:
        Reporting progress during a long task.

Functions
---------
    gen_seeds:
        Picking up bytes from os.urandom as seeds for random number
        generator.
    split_list:
        Splitting list into different groups.
"""

import time
import random
import os


class Timer(object):
    """
    Class for recording the time usage of function calls within programs.

    Attributes
    ----------
    total_time: dictionary
        overall time usage
    start_time: dictionary
        time of last self.tic() call
    end_time: dictionary
        time of last self.toc() call
    """

    def __init__(self):
        self.total_time = {}
        self.start_time = {}
        self.end_time = {}

    def tic(self, slot):
        """
        Begin tracking time usage and store it in a slot.

        :param slot: string, name of slot
        :return: None
        """
        if slot not in self.total_time.keys():
            self.total_time[slot] = 0.0
        self.start_time[slot] = time.time()

    def toc(self, slot):
        """
        Stop tracking time usage store it in a slot.

        :param slot: string, name of slot
        :return: None
        """
        if slot not in self.start_time.keys():
            raise RuntimeError("Record for slot '%s' not started!" % slot)
        else:
            self.end_time[slot] = time.time()
            self.total_time[slot] += self.end_time[slot] - \
                self.start_time[slot]

    def report_time(self):
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

    def report_total_time(self):
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

    def reset(self):
        """
        Reset self.start_time and self.end_time for next measurement.

        :return: None
        """
        self.start_time = {}
        self.end_time = {}

    def reset_total(self):
        """
        Same as self.reset, also resets self.total_time.

        :return: None
        """
        self.start_time = {}
        self.end_time = {}
        self.total_time = {}


class ProgressBar(object):
    """
    Class for reporting the progress for a time-consuming task.

    Attributes
    ----------
    num_count: integer
        total amount of tasks
        For example, if you are going to do some calculation for a vector with
        length of 1000, then set it to 1000.
    num_scales: integer
        total number of scales in the progress bar
        For example, if you want to split the whole task into 10 parts, then set
        it to 10. When one part finishes, the program will report 10% of the
        whole task has been finished.
    scale_unit: float
        amount of tasks between two adjoint scales
        See the schematic plot below for demonstration.
    next_scale: float
        the next scale waiting for counter
        If counter exceeds next_scale, then we can know one part of the task
        finish. See the schematic plot for demonstration.
    counter: integer
        counter in range of [1, num_count]
    num_scales_past: integer
        number of past scales
    percent_unit: float
        percentage of tasks between two adjoint scales

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

    def __init__(self, num_count, num_scales=10):
        self.num_count = num_count
        self.num_scales = num_scales
        self.scale_unit = num_count / num_scales
        self.next_scale = self.scale_unit
        self.counter = 0
        self.num_scales_past = 0
        self.percent_unit = 100 / num_scales

    def count(self):
        self.counter += 1
        if self.counter >= self.next_scale:
            self.num_scales_past += 1
            self.next_scale += self.scale_unit
            print("[%3d%%] finished %6d of %6d" % (self.num_scales_past * self.percent_unit,
                                                 self.counter, self.num_count))


def gen_seeds(num_seeds):
    """
    Generate a list of random numbers from /dev/urandom as seeds for Python random
    number generator.

    :param num_seeds: integer, number of seeds
    :return seeds: list of integers, seeds
    """
    num_bytes = [random.randint(1, 4) for _ in range(num_seeds)]
    byte_order = ['big' if random.randint(1, 2) == 1 else 'little' for _ in range(num_seeds)]
    #signed = [True if random.randint(1, 2) == 1 else False for _ in range(num_seeds)]
    signed = [True for _ in range(num_seeds)]
    seeds = []
    for item in zip(num_bytes, byte_order, signed):
        rand_int = int().from_bytes(os.urandom(item[0]), byteorder=item[1], signed=item[2])
        if rand_int not in seeds:
            seeds.append(rand_int)
    return seeds


def split_list(raw_list, num_group, algorithm="remainder"):
    """
    Split given list into different groups.

    Two algorithms are implemented: by the remainder of the index of each
    element divided by the number of group, or the range of index. For example,
    if we are to split the list of [0, 1, 2, 3] into two groups, by remainder
    we will get [[0, 2], [1, 3]] while by range we will get [[0, 1], [2, 3]].

    :param raw_list: list to split
    :param num_group: integer, number of groups
    :param algorithm: string, should be either "remainder" or "range"
    :return: a list containing the split list
    """
    assert num_group in range(1, len(raw_list)+1)
    assert algorithm in ("remainder", "range")
    num_element = len(raw_list)
    if algorithm == "remainder":
        list_split = [[raw_list[i] for i in range(num_element)
                      if i % num_group == k] for k in range(num_group)]
    else:
        # Get the numbers of items for each group
        num_item = [num_element // num_group for i in range(num_group)]
        for i in range(num_element % num_group):
            num_item[i] += 1
        # Divide the list according to num_item
        list_split = []
        for i in range(num_group):
            j0 = sum(num_item[:i])
            j1 = j0 + num_item[i]
            list_split.append([raw_list[j] for j in range(j0, j1)])
    return list_split


def print_banner_line(banner=None, width=80, mark="-"):
    """
    Print a banner like '#--------------- FOO ---------------' to stdout.

    The magic number 3 accounts for a '#' ahead of the banner and two spaces
    wrapping the banner text.

    Parameters
    ----------
    banner: string
        central text in the banner
    width: integer
        total width of the banner
    mark: character
        marker

    Returns
    -------
    None
    """
    num_marks_total = width - len(banner) - 3
    num_marks_left = num_marks_total // 2
    num_marks_right = num_marks_total - num_marks_left
    banner_with_marks = "#" + "".join([mark for _ in range(num_marks_left)])
    banner_with_marks += " %s " % banner
    banner_with_marks += "".join([mark for _ in range(num_marks_right)])
    print(banner_with_marks)


def print_banner_block(banner, width=80, mark="-"):
    """
    Print a banner like
    #----------------------------------#
    #               FOO                #
    #----------------------------------#
    to stdout.

    Parameters
    ----------
    banner: string
        central text in the banner
    width: integer
        total width of the banner
    mark: character
        marker

    Returns
    -------
    None
    """
    num_spaces_total = width - len(banner) - 2
    num_spaces_left = num_spaces_total // 2
    num_spaces_right = num_spaces_total - num_spaces_left
    banner_with_spaces = "#" + "".join([" " for _ in range(num_spaces_left)])
    banner_with_spaces += banner
    banner_with_spaces += "".join([" " for _ in range(num_spaces_right)]) + "#"
    border = "#" + "".join([mark for _ in range(width-2)]) + "#"
    print(border)
    print(banner_with_spaces)
    print(border)