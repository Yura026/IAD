"""
main.py — Головна точка входу для запуску програми Лабораторних робіт № 3 та № 4.
"""

import sys
import tkinter as tk
import matplotlib.pyplot as plt
from gui_lab3 import Lab3App


def main():
    root = tk.Tk()
    app = Lab3App(root)

    def on_closing():
        plt.close('all')
        root.quit()
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()