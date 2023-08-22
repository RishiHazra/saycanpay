import tkinter as tk
import time


class TextWindow:
    def __init__(self, master):
        self.master = master
        self.master.title('Text Window')
        self.master.geometry('300x200')

        self.text_widget = tk.Text(self.master, wrap=tk.WORD)
        self.text_widget.pack(expand=True, fill=tk.BOTH)

        self.display_text('Welcome to the text window!')
        self.master.after(1000, self.loop_display_text)

    def display_text(self, text):
        self.text_widget.insert(tk.END, text + '\n')
        self.text_widget.see(tk.END)

    def loop_display_text(self, line):
        self.display_text(line)
        self.master.update()
        time.sleep(0.3)


