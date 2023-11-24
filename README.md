# gravitorch-accelerate

<p align="center">
    <a href="https://github.com/durandtibo/gravitorch-accelerate/actions">
        <img alt="CI" src="https://github.com/durandtibo/gravitorch-accelerate/workflows/CI/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/gravitorch-accelerate/">
        <img alt="Documentation" src="https://github.com/durandtibo/gravitorch-accelerate/workflows/Documentation/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/gravitorch-accelerate/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/gravitorch-accelerate/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/gravitorch-accelerate/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/gravitorch-accelerate/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/gravitorch-accelerate">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/gravitorch-accelerate/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/gravitorch-accelerate/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/d7b549a77d7869aa1349/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/gravitorch-accelerate/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/d7b549a77d7869aa1349/test_coverage" />
    </a>
    <br/>
    <a href="https://pypi.org/project/gtaccelerate/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/gtaccelerate">
    </a>
    <a href="https://pypi.org/project/gtaccelerate/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/gtaccelerate.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/gtaccelerate">
    </a>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <br/>
    <a href="https://pepy.tech/project/gtaccelerate">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/gtaccelerate">
    </a>
    <a href="https://pepy.tech/project/gtaccelerate">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/gtaccelerate/month">
    </a>
    <br/>
</p>

## Overview

Plugin to use `accelerate` with `gravitorch`.

- [Documentation](https://durandtibo.github.io/gtaccelerate/)
- [Installation](#installation)
- [Contributing](#contributing)
- [API stability](#api-stability)
- [License](#license)

## Installation

We highly recommend installing
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
`gtaccelerate` can be installed from pip using the following command:

```shell
pip install gtaccelerate
```

To make the package as slim as possible, only the minimal packages required to use `gtaccelerate`
are
installed.
To include all the dependencies, you can use the following command:

```shell
pip install gtaccelerate[all]
```

Please check the [get started page](https://durandtibo.github.io/gtaccelerate/get_started) to see
how to install only some specific dependencies or other alternatives to install the library.
The following is the corresponding `gtaccelerate` versions and supported dependencies.

| `gtaccelerate` | `accelerate`   | `gravitorch`       | `torch`          | `python`      |
|----------------|----------------|--------------------|------------------|---------------|
| `main`         | `>=0.20,<0.25` | `>=0.0.23,<0.0.24` | `>=2.0.0,<2.2.0` | `>=3.9,<3.12` |

<sup>*</sup> indicates an optional dependency

## Contributing

Please check the instructions in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Suggestions and Communication

Everyone is welcome to contribute to the community.
If you have any questions or suggestions, you can
submit [Github Issues](https://github.com/durandtibo/gtaccelerate/issues).
We will reply to you as soon as possible. Thank you very much.

## API stability

:warning: While `gtaccelerate` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `gtaccelerate` to a new version will possibly break any code
that was using the old version of `gtaccelerate`.

## License

`gtaccelerate` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](LICENSE) file.
