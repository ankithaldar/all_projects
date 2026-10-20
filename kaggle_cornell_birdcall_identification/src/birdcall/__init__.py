#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cornell Birdcall Identification: two framework ports of the 1st-place solution.

The package exposes two interchangeable backends over one shared,
framework-agnostic core:

* ``pl_*`` modules are the PyTorch Lightning implementation.
* ``tf_*`` modules are the TensorFlow/Keras 3 implementation.
* Modules with neither prefix are shared and import neither framework.

The specification for every behaviour lives in the repository's ``docs/``
directory, not in this package.  See ``README.md`` for the entry points and
``tasks/plan.md`` for the architecture decisions.
'''

__version__ = '0.1.0'
