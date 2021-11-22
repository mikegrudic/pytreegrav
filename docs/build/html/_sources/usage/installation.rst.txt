.. _install: 

Installation
============

The below will help you quickly install pytreegrav.

Requirements
------------

You will need a working Python installation; we recommend installing `Anaconda <https://www.anaconda.com/download/>`_ Python version 3.x.
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

    python setup.py install

or

.. code-block:: bash

    pip install -e .

in the top level directory. The required Python packages will automatically be 
installed as well.

You can test your installation by looking for the pytreegrav 
executable built by the installation

.. code-block:: bash

    which pytreegrav

and by importing the pytreegrav Python frontend in Python

.. code-block:: python

    import pytreegrav

