import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os

from main import detect_and_analyze, OUTPUT_IMAGE, CSV_OUTPUT

APP_BG = "#0b1120"
SIDEBAR_BG = "#020617"
CARD_BG = "#111827"
ACCENT = "#3b82f6"
TEXT_PRIMARY = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
MONO_FONT = ("Consolas", 10)


class MicroGUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.current_image_tk = None
        self.image_label = None
        self.placeholder = None

        self._setup_style()
        self._build_layout()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Sidebar.TFrame", background=SIDEBAR_BG)
        style.configure("Main.TFrame", background=APP_BG)

        style.configure(
            "Title.TLabel",
            background=SIDEBAR_BG,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI Semibold", 15),
        )
        style.configure(
            "Subtitle.TLabel",
            background=SIDEBAR_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        style.configure("Card.TFrame", background=CARD_BG, relief="flat")
        style.configure(
            "CardTitle.TLabel",
            background=CARD_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardValue.TLabel",
            background=CARD_BG,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI Semibold", 15),
        )
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            font=("Segoe UI Semibold", 11),
            padding=6,
        )
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure(
            "Secondary.TButton",
            background="#1f2937",
            foreground=TEXT_PRIMARY,
            font=("Segoe UI", 10),
            padding=5,
        )
        style.map("Secondary.TButton", background=[("active", "#374151")])
        style.configure(
            "MainLabel.TLabel",
            background=APP_BG,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "Status.TLabel",
            background=SIDEBAR_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 9),
        )

    def _build_layout(self):
        self.master.configure(bg=APP_BG)
        self.master.title("Microplastics Analyzer")
        self.master.geometry("1150x700")

        # Sidebar
        sidebar = ttk.Frame(self.master, style="Sidebar.TFrame", width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(sidebar, text="Microplastics Analyzer", style="Title.TLabel").pack(
            pady=(20, 4), padx=20, anchor="w"
        )

        ttk.Label(
            sidebar,
            text="1. Upload image\n2. Run analysis\n3. View results",
            style="Subtitle.TLabel",
            justify="left",
        ).pack(padx=20, pady=(0, 14), anchor="w")

        ttk.Button(
            sidebar,
            text="Upload & Analyze",
            style="Accent.TButton",
            command=self.on_upload,
        ).pack(padx=20, pady=(0, 8), fill=tk.X)

        ttk.Button(
            sidebar,
            text="Open Latest CSV",
            style="Secondary.TButton",
            command=self.on_open_csv,
        ).pack(padx=20, pady=(0, 18), fill=tk.X)

        self.card_total = self._make_card(sidebar, "Total Particles", "--")
        self.card_area = self._make_card(sidebar, "Total Area (px)", "--")
        self.card_percent = self._make_card(sidebar, "Bowl Coverage %", "--")

        self.status_label = ttk.Label(
            sidebar,
            text="No image analyzed yet.",
            style="Status.TLabel",
        )
        self.status_label.pack(padx=20, pady=(18, 8), fill=tk.X)

        # Main area
        main = ttk.Frame(self.master, style="Main.TFrame")
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        top_bar = ttk.Frame(main, style="Main.TFrame")
        top_bar.pack(fill=tk.X, padx=20, pady=(14, 2))

        ttk.Label(top_bar, text="Annotated Output", style="MainLabel.TLabel").pack(
            side=tk.LEFT
        )

        self.image_frame = tk.Frame(main, bg="#020617")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.placeholder = tk.Label(
            self.image_frame,
            text="Upload an image to see the annotated result here.",
            bg="#020617",
            fg=TEXT_MUTED,
            font=("Segoe UI", 11),
        )
        self.placeholder.pack(expand=True)

        log_frame = ttk.Frame(main, style="Main.TFrame", height=150)
        log_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        ttk.Label(log_frame, text="Analysis Log", style="MainLabel.TLabel").pack(
            anchor="w"
        )

        self.log_text = tk.Text(
            log_frame,
            height=6,
            bg="#020617",
            fg=TEXT_MUTED,
            insertbackground=TEXT_PRIMARY,
            bd=0,
            relief=tk.FLAT,
            font=MONO_FONT,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def _make_card(self, parent, title, value):
        outer = ttk.Frame(parent, style="Sidebar.TFrame")
        outer.pack(fill=tk.X, pady=4, padx=20)

        frame = ttk.Frame(outer, style="Card.TFrame")
        frame.pack(fill=tk.X)

        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(
            anchor="w", padx=10, pady=(6, 0)
        )

        val_lbl = ttk.Label(frame, text=value, style="CardValue.TLabel")
        val_lbl.pack(anchor="w", padx=10, pady=(0, 6))

        return val_lbl

    # ---------- EVENTS ----------
    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not path:
            return

        try:
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert(tk.END, f"Analyzing: {path}\n")

            detect_and_analyze(path)

            # reload latest CSV_OUTPUT value from main
            from main import CSV_OUTPUT as LATEST_CSV

            self.log_text.insert(tk.END, f"Annotated image: {OUTPUT_IMAGE}\n")
            self.log_text.insert(tk.END, f"CSV: {LATEST_CSV}\n")
            self.status_label.config(text="Last run: OK")

            if os.path.exists(OUTPUT_IMAGE):
                self.show_image(OUTPUT_IMAGE)
            else:
                messagebox.showwarning(
                    "Missing output",
                    f"Expected '{OUTPUT_IMAGE}' not found.",
                )

            self.update_stats_from_csv(LATEST_CSV)
        except Exception as e:
            self.status_label.config(text="Last run: ERROR")
            messagebox.showerror("Error", str(e))

    def on_open_csv(self):
        from main import CSV_OUTPUT as LATEST_CSV

        if os.path.exists(LATEST_CSV):
            os.startfile(LATEST_CSV)
        else:
            messagebox.showinfo("Info", "CSV not found. Run an analysis first.")

    def show_image(self, path):
        if self.placeholder:
            self.placeholder.destroy()
            self.placeholder = None

        img = Image.open(path)
        frame_w = self.image_frame.winfo_width() or 800
        frame_h = self.image_frame.winfo_height() or 450
        w, h = img.size
        scale = min(frame_w / w, frame_h / h, 1.0)
        img = img.resize((int(w * scale), int(h * scale)))

        self.current_image_tk = ImageTk.PhotoImage(img)

        if self.image_label is None:
            self.image_label = tk.Label(
                self.image_frame, image=self.current_image_tk, bg="#020617"
            )
            self.image_label.pack(expand=True)
        else:
            self.image_label.configure(image=self.current_image_tk)
            self.image_label.image = self.current_image_tk

    def update_stats_from_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return

        total_particles = "-"
        total_area = "-"
        percent = "-"

        try:
            with open(csv_path, "r") as f:
                for line in f:
                    if line.startswith("Total Particles"):
                        parts = line.strip().split(",")
                        if len(parts) > 1:
                            total_particles = parts[1]
                    elif line.startswith("Total Area(px)"):
                        parts = line.strip().split(",")
                        if len(parts) > 1:
                            total_area = parts[1]
                    elif line.startswith("Percent Bowl Area"):
                        parts = line.strip().split(",")
                        if len(parts) > 1:
                            percent = parts[1].replace("%", "")
        except Exception:
            pass

        self.card_total.config(text=str(total_particles))
        self.card_area.config(text=str(total_area))
        self.card_percent.config(text=str(percent))


if __name__ == "__main__":
    root = tk.Tk()
    app = MicroGUI(root)
    root.mainloop()
