"""Geometry utilities: minimum-image coordinate math and graph construction.

Import submodules explicitly (``basinflow.geometry.graph`` etc.).
This package deliberately re-exports nothing: the graph builder pulls in torch,
so an eager re-export would make torch a hard dependency of ``basinflow.geometry``.
"""
