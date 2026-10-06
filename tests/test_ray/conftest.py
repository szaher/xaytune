"""One real Ray head per session, however many modules use it.

Registered here, once: a session fixture imported into several test modules
is a separate fixture in each, and each would start its own head.
"""

from tests.test_ray.ray_support import ray_head

__all__ = ["ray_head"]
