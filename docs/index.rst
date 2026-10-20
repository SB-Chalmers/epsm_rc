EPSM RC
=======

RC reduces an explicitly authored Core Building to an hourly thermal network.
Core owns physical geometry, constructions and weather inspection. RC owns the
reduction assumptions, thermal calculation and its output meaning.

Start with the physical Building route below. The standalone API remains useful
for explicitly supplied numerical experiments and supports Python 3.10 without
Core. The shared Building route requires Python 3.11 or newer and the coordinated
``platform`` extra.

.. toctree::
   :maxdepth: 2

   quickstart
   api
   standalone
   migration

Software checks establish consistency and specified behavior. They do not certify
building energy performance or establish prediction accuracy against measurements.
