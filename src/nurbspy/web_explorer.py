"""Compatibility import for the experimental embeddable editor.

New integrations should import GeometryEditor from nurbspy.visualization.
"""
from .visualization.dash import GeometryEditor as NurbsExplorer

__all__ = ["NurbsExplorer"]
