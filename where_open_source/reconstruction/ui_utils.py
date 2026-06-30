import tkinter as tk
from tkinter import font


def get_chinese_font() -> str:
    """Select an available CJK font for better Chinese text rendering."""
    root = tk.Tk()
    root.withdraw()
    available_fonts = list(font.families())
    root.destroy()

    priority_fonts = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Noto Sans Mono CJK SC",
        "SimHei",
    ]
    for f in priority_fonts:
        if f in available_fonts:
            return f
    return "DejaVu Sans"
