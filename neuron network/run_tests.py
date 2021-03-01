import unittest
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests')
    JSONTestRunner(visibility='after_published').run(suite)
    # students shouldn't be able to see results until grades are published
    #JSONTestRunner(visibility='after_due_date').run(suite)
