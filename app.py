"""Hugging Face Spaces / local entry point. Spaces runs this file and finds
``demo``; locally, ``python app.py`` launches the GUI.
"""
from gui.app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
