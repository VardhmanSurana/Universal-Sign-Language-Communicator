import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import sys

# --- Logic to run scripts ---
def run_script(folder_name, script_name, title):
    full_path = os.path.join(folder_name, script_name)

    if os.path.exists(full_path):
        try:
            subprocess.Popen([sys.executable, script_name], cwd=folder_name)
            messagebox.showinfo(title, f"✅ {title} launched successfully!")
        except FileNotFoundError:
            messagebox.showerror("Python Not Found", "⚠️ Python is not in your system PATH.")
        except Exception as e:
            messagebox.showerror("Execution Error", f"❌ An unexpected error occurred:\n\n{e}")
    else:
        messagebox.showerror("File Not Found", f"📁 Couldn't find:\n{full_path}")

# --- GUI Layout ---
def create_gui():
    root = tk.Tk()
    root.title("🧠 Sign Language Translator")
    root.geometry("420x300")
    root.configure(bg="#f2f2f2")

    # Title
    title_label = tk.Label(root, text="🧠 Sign Language Translator", font=("Helvetica", 18, "bold"), bg="#f2f2f2")
    title_label.pack(pady=20)

    # Buttons Frame
    btn_frame = tk.Frame(root, bg="#f2f2f2")
    btn_frame.pack(pady=10)

    # Buttons
    speech_btn = tk.Button(btn_frame, text="🗣️  Speech to Sign", font=("Helvetica", 14), width=20, bg="#4CAF50", fg="white",
                           command=lambda: run_script(".", "Speech-to-Sign/abc.py", "Speech-to-Sign"))  # Now runs from current folder
    speech_btn.grid(row=0, column=0, pady=10)

    sign_btn = tk.Button(btn_frame, text="🤟  Sign to Speech", font=("Helvetica", 14), width=20, bg="#2196F3", fg="white",
                         command=lambda: run_script("Sign-to-Speech", "main.py", "Sign-to-Speech"))
    sign_btn.grid(row=1, column=0, pady=10)

    # Exit button
    exit_btn = tk.Button(root, text="❌ Exit", font=("Helvetica", 12), width=15, bg="#f44336", fg="white", command=root.quit)
    exit_btn.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
