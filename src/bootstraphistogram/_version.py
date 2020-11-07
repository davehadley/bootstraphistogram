def version(packagename: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(packagename)  # type: ignore
    except ImportError:
        import pkg_resources

        return pkg_resources.get_distribution(packagename).version
    except Exception:
        return "unknown"
