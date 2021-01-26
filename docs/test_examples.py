import glob
import os
from subprocess import check_call

import pytest


def _getexamplefiles():
    dirname = os.path.dirname(os.path.abspath(__file__))
    return [
        fname
        for fname in glob.glob(os.sep.join((dirname, "*", "*.py")))
        if __file__ not in fname
    ]

@pytest.mark.xfail
@pytest.mark.parametrize("examplefile", _getexamplefiles())
def test_example(examplefile):
    check_call(["python", examplefile])
