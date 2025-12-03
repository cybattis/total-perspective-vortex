import tkinter
from tkinter import ttk
import sv_ttk

BACKGROUND = "#808080"
TEXT = "Hello world"


def create_app() -> tkinter.Tk:
    root_folder = __file__.rsplit("/", 1)[0]    
    print("Root folder:", root_folder)
    
    root = tkinter.Tk()
    sv_ttk.set_theme("dark")

    root.title("Total Perspective Vortex")
    root.geometry("1280x720")

    menubar = tkinter.Menu(root)
    file_menu = tkinter.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Exit", command=root.destroy)
    menubar.add_cascade(label="File", menu=file_menu)

    view_menu = tkinter.Menu(menubar, tearoff=0)
    view_menu.add_command(label="Toggle Theme", command=sv_ttk.toggle_theme)
    menubar.add_cascade(label="View", menu=view_menu)

    root.config(menu=menubar)
    
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2)

    config_frame = ttk.Frame(root, padding=20, borderwidth=2, relief="ridge")
    config_frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(config_frame, text="Configuration").pack(anchor="w")

    display_frame = ttk.Frame(root, padding=20, borderwidth=2, relief="ridge")
    display_frame.grid(row=0, column=1, sticky="nsew")
    ttk.Label(display_frame, text="Visual Display Area").pack(anchor="center", expand=True)
    
    return root


if __name__ == "__main__":
    app = create_app()
    app.mainloop()