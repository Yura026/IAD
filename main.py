import sys
import tkinter as tk
import matplotlib.pyplot as plt
from gui import DataMiningApp

def on_closing(root):
    # Закриваємо всі фігури matplotlib, щоб вони не тримали процес
    plt.close('all')
    root.quit()
    root.destroy()
    sys.exit(0)

def main():
    root = tk.Tk()
    app = DataMiningApp(root)
    # Прив'язуємо чисте завершення до хрестика вікна
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    root.mainloop()

if __name__ == "__main__":
    main()