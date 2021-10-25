"""Functions and classes for simplifying unit tests."""

from io import StringIO
from unittest.mock import patch

import numpy as np


class TestHelper:
    """
    Helper class that makes unittest easier.
    """
    def __init__(self, tester):
        """
        :param tester: unittest.TestCase object
            instance of unittest.TestCase
        """
        self.tester = tester

    def test_equal_array(self, array1, array2, almost=False):
        """
        Checks if two arrays are equal to each other.

        :param array1: numpy array, left operand of comparison
        :param array2: numpy array, right operand of comparison
        :return: None.
        """
        diff = np.sum(np.abs(array1 - array2)).item(0)
        if not almost:
            self.tester.assertEqual(diff, 0)
        else:
            self.tester.assertAlmostEqual(diff, 0.0)

    def test_raise(self, func, exception, message=None):
        """
        Tests if expected exception is raised during an operation.

        :param func: function object
            wrapper function over the operation to test
        :param exception: class object
            category of exception to test
        :param message: regex
            expected exception message
        :return:
        """
        with self.tester.assertRaises(exception) as cm:
            func()
        if message is not None:
            self.tester.assertRegex(str(cm.exception), message)

    def test_no_raise(self, func, exception):
        """
        Tests if expected exception is not raised during an operation.

        :param func: function object
            wrapper function over the operation to test
        :param exception: class object
            category of exception to test
        :return: None
        """
        try:
            func()
        except exception:
            status = True
        else:
            status = False
        self.tester.assertFalse(status)

    def test_stdout(self, func, message):
        """
        Test if the output contain given message.
        :param func: function object
            wrapper function over the operation to test
        :param message: list of regex
            reference message with which to compare
            each regex corresponds to one line of output
        :return: None
        """
        with patch('sys.stdout', new=StringIO()) as fake_out:
            func()
        output = [out for out in fake_out.getvalue().split("\n") if out != ""]
        for i, msg in enumerate(message):
            self.tester.assertRegex(output[i], msg)
