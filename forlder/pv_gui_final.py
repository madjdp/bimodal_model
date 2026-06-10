# Madj modèle final ===
#Email: madjstar10@gmail.com

import os
import json
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io

# ------------- CONFIG -------------
MODEL_FILENAME = "analyse_bimodal/modele_bimodal_solaire_keras_madj.keras"
IMG_SIZE = (224, 224)
DISPLAY_MAX = (480, 480)
DEFAULT_CLASS_NAMES = ['normal', 'diode', 'hotspot']
# --- STYLE DARK MODE ---
BG_COLOR = "#2e3440"
FG_COLOR = "#d8dee9"
ACCENT_COLOR = "#88c0d0"
BTN_BG = "#4c566a"
BTN_HOVER_BG = "#81a1c1"
TXT_BG = "#3b4252"
# -----------------------------------


def style_button(btn):
    btn.configure(
        bg=BTN_BG,
        fg=FG_COLOR,
        activebackground=BTN_HOVER_BG,
        activeforeground=FG_COLOR,
        bd=0,
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        cursor="hand2",
        padx=12,
        pady=6,
    )

    def on_enter(e):
        btn['bg'] = BTN_HOVER_BG

    def on_leave(e):
        btn['bg'] = BTN_BG

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)


def try_load_class_names(json_path="class_indices.json"):
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                d = json.load(f)
            inv = {}
            for k, v in d.items():
                try:
                    inv[int(v)] = k
                except Exception:
                    inv[k] = v
            return [inv[i] for i in sorted(inv.keys())]
        except Exception as e:
            print("Impossible de lire class_indices.json :", e)
    return None


def load_model_auto():
    """Charge uniquement le modèle défini dans MODEL_FILENAME"""
    if os.path.exists(MODEL_FILENAME):
        try:
            model = load_model(MODEL_FILENAME, compile=False)
            print("Modèle chargé :", MODEL_FILENAME)
            return model, MODEL_FILENAME
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle défini : {e}")
            return None, None
    else:
        messagebox.showerror("Erreur", f"Le fichier de modèle spécifié n'existe pas :\n{MODEL_FILENAME}")
        return None, None


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img, arr


def load_electric_data(path):
    """
    Charge les données électriques depuis un fichier texte ou CSV simple.
    Le format attendu : une ligne avec les features séparées par virgules ou espaces.
    """
    try:
        with open(path, "r") as f:
            line = f.readline().strip()
        parts = line.split(",") if "," in line else line.split()
        arr = np.array([float(x) for x in parts], dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        messagebox.showerror("Erreur données électriques", f"Impossible de charger les données électriques : {e}")
        return None


def predict(model, img_arr, electric_arr):
    preds = model.predict([img_arr, electric_arr])
    probs = preds[0]
    idx = int(np.argmax(probs))
    return idx, probs


def export_pdf_file(image_path, overlay_pil, pred_class, prob, class_probs,
                    model_name, save_path, class_names, hist_img=None, electric_data=None):
    """Crée un rapport PDF propre et professionnel (gère les sauts de page pour éviter chevauchement)"""
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4
    margin = 50

    def draw_header():
        y0 = height - margin
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, y0, "Rapport de prédiction - SEN Diagnostic PV")
        y = y0 - 22
        c.setFont("Helvetica", 9)
        c.drawCentredString(width / 2, y, f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 26
        c.setLineWidth(0.5)
        c.line(margin, y + 10, width - margin, y + 10)
        return y

    def ensure_space(y, needed):
        # si l'espace restant est insuffisant, nouvelle page et réaffichage de l'en-tête
        if y - needed < margin:
            c.showPage()
            return draw_header()
        return y

    y = draw_header()

    # --- Image principale ---
    try:
        orig = Image.open(image_path).convert("RGB")
    except Exception:
        orig = overlay_pil.copy() if overlay_pil is not None else Image.new("RGB", (IMG_SIZE[0], IMG_SIZE[1]), (255,255,255))

    # calcul taille en conservant marges
    w_img = width - 2 * margin
    h_img = 260
    orig.thumbnail((w_img, h_img))
    img_w, img_h = orig.size

    y = ensure_space(y, img_h + 10)
    c.drawImage(ImageReader(orig), margin + (w_img - img_w) / 2, y - img_h, width=img_w, height=img_h)
    y -= img_h + 20

    # --- Graphique des probabilités ---
    if hist_img:
        hist_w = width - 2 * margin
        hist_h = 220  # hauteur suffisante pour la netteté
        if not isinstance(hist_img, Image.Image):
            try:
                hist_img = Image.open(hist_img)
            except Exception:
                hist_img = None
        if hist_img:
            hist_img = hist_img.copy()
            hist_img.thumbnail((hist_w, hist_h), Image.LANCZOS)
            hw, hh = hist_img.size
            y = ensure_space(y, hh + 10)
            c.drawImage(ImageReader(hist_img), margin + (hist_w - hw) / 2, y - hh, width=hw, height=hh)
            y -= hh + 18

    # --- Résultat résumé ---
    y = ensure_space(y, 60)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Résumé de la prédiction :")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(margin + 10, y, f"Classe prédite : {pred_class}")
    y -= 14
    c.drawString(margin + 10, y, f"Probabilité : {prob*100:.2f}%")
    y -= 18

    # --- Probabilités détail ---
    y = ensure_space(y, 14 * (len(class_probs) + 2))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Probabilités par classe :")
    y -= 14
    c.setFont("Helvetica", 10)
    table_col_x = margin + 10
    for i, p in enumerate(class_probs):
        cls_name = class_names[i] if i < len(class_names) else f"Classe {i}"
        line = f"- {cls_name.ljust(12)} : {p*100:6.2f} %"
        c.drawString(table_col_x, y, line)
        y -= 12
    y -= 8

    # --- Données électriques ---
    if electric_data is not None:
        flat = electric_data.flatten()
        elec_labels = ["Voc", "Isc", "Vmp", "Imp", "Pmax", "Vmp/Voc", "Imp/Voc"]
        rows_needed = max(len(flat), len(elec_labels)) + 3  # en-têtes + marges
        y = ensure_space(y, 14 * rows_needed)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Données électriques fournies :")
        y -= 18

        c.setFont("Helvetica-Bold", 10)
        col1_x = margin + 10
        col2_x = margin + 200
        c.drawString(col1_x, y, "Paramètre")
        c.drawString(col2_x, y, "Valeur")
        y -= 14
        c.setFont("Helvetica", 10)
        for i in range(max(len(flat), len(elec_labels))):
            label = elec_labels[i] if i < len(elec_labels) else f"Feature {i+1}"
            val = flat[i] if i < len(flat) else float('nan')
            c.drawString(col1_x, y, f"{label}")
            c.drawRightString(col2_x + 80, y, f"{val:.4f}")
            y -= 12
        y -= 8

    # --- Interprétation ---
    # calcule espace nécessaire approximatif pour le bloc d'interprétation
    interp_lines = 6
    y = ensure_space(y, 14 * interp_lines)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Interprétation:")
    y -= 16
    c.setFont("Helvetica", 10)
    pred_lower = str(pred_class).lower()
    if "normal" in pred_lower:
        texte = ("Le module photovoltaïque est jugé en bon état de fonctionnement selon les données "
                 "thermiques et électriques fournies. Aucune action immédiate requise.")
    elif "diode" in pred_lower:
        texte = ("Anomalie possible liée à une diode de dérivation. Une inspection physique est recommandée.")
    elif "hotspot" in pred_lower:
        texte = ("Hotspot détecté : surchauffe localisée. Contrôler la zone identifiée et envisager "
                 "un remplacement ou nettoyage du module.")
    else:
        texte = ("Anomalie détectée. Une inspection plus approfondie est recommandée.")
    # wrap texte manuellement sur environ 100 chars
    max_chars = 100
    lines = []
    while texte:
        lines.append(texte[:max_chars])
        texte = texte[max_chars:]
    for ln in lines:
        # si on manque d'espace pour la prochaine ligne, nouvelle page
        if y - 14 < margin:
            c.showPage()
            y = draw_header()
            c.setFont("Helvetica", 10)
        c.drawString(margin + 10, y, ln)
        y -= 12
    y -= 10

    # --- Pied de page ---
    # si pas assez d'espace, nouvelle page pour footer
    if y < margin + 40:
        c.showPage()
        y = draw_header()
    c.setFont("Helvetica-Oblique", 9)
    c.setFillGray(0.45)
    c.drawCentredString(width / 2, 28, "Généré automatiquement par SEN Diagnostic PV © " + str(datetime.now().year))

    c.save()
    return save_path


class PVApp:
    def __init__(self, root):
        self.root = root
        root.title("SEN Diagnostic PV")
        root.geometry("1400x820")
        root.configure(bg=BG_COLOR)

        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(0, weight=1)

        img_container = tk.Frame(root, bd=2, relief="sunken", bg=BG_COLOR)
        img_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        img_container.grid_rowconfigure(0, weight=1)
        img_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(img_container, background=BG_COLOR, highlightthickness=0)
        self.scroll_y = tk.Scrollbar(img_container, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(img_container, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        self.scroll_y.grid(row=0, column=1, sticky="ns")
        self.scroll_x.grid(row=1, column=0, sticky="ew")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.img_frame = tk.Frame(self.canvas, bg=BG_COLOR)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.img_frame, anchor="nw")
        self.img_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.img_container = tk.Frame(self.img_frame, padx=20, pady=20, bg=BG_COLOR)
        self.img_container.pack(expand=True, fill=tk.BOTH)
        self.img_label = tk.Label(self.img_container, text="Aucune image", bg=BG_COLOR, fg=FG_COLOR,
                                  font=("Segoe UI", 14, "italic"))
        self.img_label.pack(expand=True)

        right = tk.Frame(root, bg=BG_COLOR)
        right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctrl = tk.Frame(right, bg=BG_COLOR)
        ctrl.grid(row=0, column=0, sticky="ew", pady=4)

        self.load_img_btn = tk.Button(ctrl, text="Charger image", command=self.on_load_image)
        self.load_elec_btn = tk.Button(ctrl, text="Charger données électriques", command=self.on_load_electric)
        self.predict_btn = tk.Button(ctrl, text="Prédire", command=self.on_predict, state=tk.DISABLED)
        self.pdf_btn = tk.Button(ctrl, text="Exporter PDF", command=self.on_export_pdf, state=tk.DISABLED)

        for btn in [self.load_img_btn, self.load_elec_btn, self.predict_btn, self.pdf_btn]:
            btn.pack(side=tk.LEFT, padx=4)
            style_button(btn)

        self.result_text = tk.Text(right, width=48, height=10, bg=TXT_BG, fg=FG_COLOR,
                                   font=("Consolas", 11), bd=0, relief="flat", insertbackground=ACCENT_COLOR, wrap="word")
        self.result_text.grid(row=1, column=0, pady=8, sticky="nsew")
        self.result_text.insert("1.0", "Aucune prédiction — chargez image et données électriques")
        self.result_text.config(state=tk.DISABLED)

        fig_frame = tk.Frame(right, bg=BG_COLOR)
        fig_frame.grid(row=2, column=0, pady=6, sticky="nsew")
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(5.5, 3.5))
        self.ax.set_title("Probabilités par classe", color=FG_COLOR)
        self.ax.tick_params(axis='x', colors=FG_COLOR)
        self.ax.tick_params(axis='y', colors=FG_COLOR)
        self.fig.patch.set_facecolor(BG_COLOR)

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas_fig.get_tk_widget().configure(bg=BG_COLOR)
        self.canvas_fig.get_tk_widget().pack_forget()

        # placeholders / état
        self.model = None
        self.model_path = None
        self.class_names = try_load_class_names() or DEFAULT_CLASS_NAMES
        self.current_image_path = None
        self.current_orig_pil = None
        self.current_img_arr = None
        self.current_overlay = None
        self.last_pred = None
        self.tk_img = None
        self.hist_img = None
        self.electric_data = None
        self.expected_elec_dim = None

        # charger le modèle
        try:
            self.model, self.model_path = load_model_auto()
            if self.model is None:
                
                pass
            else:
                shapes = [inp.shape for inp in self.model.inputs]
                print("Shapes des entrées du modèle :", shapes)
                try:
                    self.expected_elec_dim = int(self.model.inputs[1].shape[-1])
                    print(f"Dimension électrique attendue : {self.expected_elec_dim}")
                except Exception:
                    messagebox.showwarning("Attention", "Le modèle chargé semble ne pas être bimodal.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement du modèle : {e}")

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def on_load_image(self):
        path = filedialog.askopenfilename(title="Choisir une image thermographique",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("Tous fichiers", "*.*")])
        if path:
            self.current_image_path = path
            self.current_orig_pil, self.current_img_arr = preprocess_image(path)
            self.show_image(self.current_orig_pil)
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Image chargée.\nCharge maintenant les données électriques.")
            self.result_text.config(state=tk.DISABLED)
            self.predict_btn.config(state=tk.DISABLED)
            self.pdf_btn.config(state=tk.DISABLED)

    def show_image(self, pil_img):
        pil_img.thumbnail(DISPLAY_MAX)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.tk_img, text="")

    def on_load_electric(self):
        path = filedialog.askopenfilename(title="Charger données électriques",
                                          filetypes=[("Fichiers texte", "*.txt *.csv"), ("Tous fichiers", "*.*")])
        if path:
            data = load_electric_data(path)
            if data is not None:
                self.electric_data = data
                self.result_text.config(state=tk.NORMAL)
                self.result_text.insert(tk.END, f"\nDonnées électriques chargées : {data.shape[1]} features.")
                self.result_text.config(state=tk.DISABLED)
                if self.current_image_path:
                    self.predict_btn.config(state=tk.NORMAL)

    def on_predict(self):
        if not self.model:
            messagebox.showerror("Erreur", "Aucun modèle chargé.")
            return
        if self.current_img_arr is None or self.electric_data is None:
            messagebox.showwarning("Attention", "Charge d'abord une image ET des données électriques.")
            return

        # Vérification dimension données électriques
        if self.expected_elec_dim is not None and self.electric_data.shape[1] != self.expected_elec_dim:
            if not messagebox.askyesno("Confirmation", f"Données électriques ont {self.electric_data.shape[1]} features, alors que le modèle attend {self.expected_elec_dim}. Continuer ?"):
                return

        idx, probs = predict(self.model, self.current_img_arr, self.electric_data)
        pred_class = self.class_names[idx] if idx < len(self.class_names) else f"Classe {idx}"
        prob = float(probs[idx])

        self.last_pred = (pred_class, prob, probs)

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Prédiction : {pred_class} (probabilité : {prob*100:.2f}%)\n\nProbabilités détaillées:\n")
        for i, p in enumerate(probs):
            cls_name = self.class_names[i] if i < len(self.class_names) else f"Classe {i}"
            self.result_text.insert(tk.END, f"{cls_name}: {p*100:.2f}%\n")
        self.result_text.config(state=tk.DISABLED)

        # Afficher graphique
        self.ax.clear()
        colors = ["#88c0d0", "#a3be8c", "#ebcb8b", "#bf616a", "#b48ead"]
        bar_colors = colors[:len(self.class_names)]
        bars = self.ax.bar(self.class_names, probs, color=bar_colors)
        self.ax.set_ylim([0, 1])
        self.ax.set_title("Probabilités par classe", color=FG_COLOR)
        self.ax.tick_params(axis='x', colors=FG_COLOR, rotation=25)
        self.ax.tick_params(axis='y', colors=FG_COLOR)
        self.fig.patch.set_facecolor(BG_COLOR)

        for bar, probv in zip(bars, probs):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{probv*100:.1f}%",
                         ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        self.pdf_btn.config(state=tk.NORMAL)

    def on_export_pdf(self):
        if not self.last_pred:
            messagebox.showwarning("Avertissement", "Aucune prédiction à exporter.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                 filetypes=[("PDF files", "*.pdf")],
                                                 title="Enregistrer rapport PDF")
        if save_path:
            pred_class, prob, probs = self.last_pred

            overlay_img = self.current_orig_pil if self.current_orig_pil else None

            # récupérer image du plot (histogramme/probabilités)
            hist_buf = io.BytesIO()
            try:
                self.fig.savefig(hist_buf, format='png', bbox_inches='tight')
                hist_buf.seek(0)
                hist_img = Image.open(hist_buf)
            except Exception:
                hist_img = None

            model_name = os.path.basename(self.model_path) if self.model_path else "Modèle bimodal"

            try:
                export_pdf_file(
                    image_path=self.current_image_path,
                    overlay_pil=overlay_img,
                    pred_class=pred_class,
                    prob=prob,
                    class_probs=probs,
                    model_name=model_name,
                    save_path=save_path,
                    class_names=self.class_names,
                    hist_img=hist_img,
                    electric_data=self.electric_data
                )
                messagebox.showinfo("Succès", f"Rapport PDF enregistré dans:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'exporter le PDF : {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PVApp(root)
    root.mainloop()
