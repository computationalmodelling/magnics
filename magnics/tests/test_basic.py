import sys

def test_python_version():
    assert sys.version_info.major >= (3, 5)

def test_arithmetic():
    assert 2 + 2 == 4
