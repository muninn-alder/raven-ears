"""Check backend versions before Transformers can hide an incompatible Torch."""
from importlib import metadata


class ModelVersionError(ImportError):
    remedy_first = True


def check_transformers_torch():
    from packaging.requirements import Requirement
    from packaging.version import Version

    try:
        transformers = metadata.version("transformers")
        torch = metadata.version("torch")
        requirements = metadata.requires("transformers") or []
    except metadata.PackageNotFoundError:
        return  # Let the backend report the missing package at its usual boundary.
    for line in requirements:
        requirement = Requirement(line)
        if requirement.name != "torch" or (requirement.marker and not requirement.marker.evaluate({"extra": "torch"})):
            continue
        if Version(torch) not in requirement.specifier:
            raise ModelVersionError(f"your torch {torch} does not match transformers {transformers} "
                                    f"(requires torch{requirement.specifier}); see INSTALL.md")
