import ripple


def test_imports_with_version():
    assert isinstance(ripple.__version__, str)
