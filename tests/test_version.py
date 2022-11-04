import bootstraphistogram


def _package_meta_data_version(packagename: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(packagename)
    except ImportError:
        import pkg_resources

        return pkg_resources.get_distribution(packagename).version


def testversion() -> None:
    assert (
        bootstraphistogram.__version__
        == "0.11.0"
        == _package_meta_data_version(bootstraphistogram.__name__)
    )
