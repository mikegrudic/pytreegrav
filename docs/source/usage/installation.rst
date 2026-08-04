.. _install: 

Installation
============

The below will help you quickly install pytreegrav.

Requirements
------------

You will need a working Python 3.x installation; we recommend installing `Anaconda <https://www.anaconda.com/download/>`_ Python version 3.x.
You will also need to install the following packages:

    * numpy

    * numba

Installing the latest stable release
------------------------------------

Install the latest stable release with

.. code-block:: bash

    pip install pytreegrav

This is the preferred way to install pytreegrav as it will
automatically install the necessary requirements and put Pytreegrav
into your :code:`${PYTHONPATH}` environment variable so you can 
import it.

Install from source
-------------------

Alternatively, you can install the latest version directly from the most up-to-date version
of the source-code by cloning/forking the GitHub repository 

.. code-block:: bash

    git clone https://github.com/mikegrudic/pytreegrav.git


Once you have the source, you can build pytreegrav (and add it to your environment)
by executing

.. code-block:: bash

    pip install .

or, if you want your edits to the source to take effect without reinstalling,

.. code-block:: bash

    pip install -e .

in the top level directory. The required Python packages will automatically be
installed as well.

You can test your installation by importing the pytreegrav Python frontend in Python

.. code-block:: python

    import pytreegrav

Optional: GPU acceleration
--------------------------

Column density, monopole tree gravity and brute-force gravity have an optional CUDA backend. On clustered
production data, against 32 CPU threads: column density ~11x from a single ``device="cuda"`` call, gravity
~4x that way but ~32x with the tree held resident on the device, and ~40x for brute force. Install the
extra with

.. code-block:: bash

    pip install pytreegrav[cuda]

See :ref:`cuda` for usage and caveats. This is optional in the strict sense: nothing in pytreegrav
imports it unless you ask for it, so a CPU-only install is unaffected.

Testing
-------

To test that the tree solver is working correctly, run

.. code-block:: bash

    pytest

from the root directory of the package. This will run a basic test problem comparing the acceleration and potential from the tree and brute force solvers respectively, and check that the answers are within the expected tolerance.
