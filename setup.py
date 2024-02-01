from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    lines = f.read().split("\n")
    requirements = []
    for line in lines:
        if line.startswith("git+"):
            link, package = line.split("#egg=")
            requirements.append(f"{package} @ {link}#{package}")
        else:
            requirements.append(line)

setup(
    name="adex_gym",
    version="1.0.0",
    long_description=long_description,
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "pytest",
            "docker",
        ]
    },
    packages=find_packages(exclude=["maps", "cache", "scenarios", "agents"])
)
