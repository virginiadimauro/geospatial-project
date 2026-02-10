"""
Geospatial Analysis Pipeline
Package for Madrid Airbnb data cleaning, enrichment, and spatial integration.
"""

__version__ = "1.0.0"
__author__ = "Virginia Di Mauro"

# Lazy imports to avoid circular dependencies and long startup times
# Import as needed in code

__all__ = ["config", "io", "cleaning", "reviews", "spatial", "qc"]
