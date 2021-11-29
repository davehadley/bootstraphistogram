import bootstraphistogram


def _package_meta_data_version(packagename: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(packagename)  # type: ignore
    except ImportError:
        import pkg_resources

        return pkg_resources.get_distribution(packagename).version
    except Exception:
        return "unknown"


def testversion():
    assert (
        bootstraphistogram.__version__
        == "0.8.0"
        == _package_meta_data_version(bootstraphistogram.__name__)
    )
