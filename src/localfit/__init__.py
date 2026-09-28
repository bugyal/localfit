"""localfit — what will run on your hardware, and at what speed.

The question this answers is asked thousands of times a week in per-GPU hobbyist
threads and answered by hand every time: *will this model run on my card, at what
quantisation, how fast, and why is my two-GPU load lopsided?*

Two things make that answerable offline, before downloading a single file:

1. Memory is arithmetic. Weights, KV cache and compute buffers are all
   computable from the model config, the quantisation and the context length.
2. Decode speed is bounded by memory bandwidth. A token needs every active
   weight read once, so ``bandwidth / bytes-per-token`` is a hard ceiling; the
   achievable fraction of it depends on the architecture and the quantisation
   kernel, which is where measured community data enters.

Every number this package prints is either arithmetic from published specs or a
quoted measurement with its source. Where the two disagree, the measurement wins
but the bound is still shown, so you can see which one is limiting.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
