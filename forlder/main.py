# pv_app_improved_full.py
import os
import json
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tempfile

# ------------- CONFIG -------------
MODEL_FILENAME = "efficientnetv2_pv_model.keras"
IMG_SIZE = (224, 224)           # taille attendue par ton modèle
DISPLAY_MAX = (480, 480)        # taille max d'affichage
DEFAULT_CLASS_NAMES = ["anormal", "diode", "hotspot", "normal"]
# -----------------------------------

def try_load_class_names(json_path="class_indices.json"):
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                d = json.load(f)
            # si c'est mapping name->index on inverse
            inv = {}
            for k, v in d.items():
                try:
                    inv[int(v)] = k
                except:
                    inv[k] = v
            return [inv[i] for i in sorted(inv.keys())]
        except Exception as e:
            print("Impossible de lire class_indices.json :", e)
    return None

def load_model_auto():
    """Charge automatiquement le modèle .keras/.h5 si présent, sinon demande à l'utilisateur."""
    candidates = [MODEL_FILENAME, "efficientnetv2_pv_model.h5", "efficientnetv2_pv_model_finetuned.keras"]
    for c in candidates:
        if os.path.exists(c):
            try:
                m = load_model(c, compile=False)
                print("Modèle chargé :", c)
                return m, c
            except Exception as e:
                print("Impossible de charger", c, ":", e)
    # demander chemin
    path = filedialog.askopenfilename(title="Choisir un modèle (.keras/.h5)",
                                      filetypes=[("Keras model", "*.keras *.h5 *.hdf5"), ("All", "*.*")])
    if path:
        try:
            m = load_model(path, compile=False)
            print("Modèle chargé :", path)
            return m, path
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle sélectionné : {e}")
    return None, None

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img, arr

def predict(model, img_arr):
    preds = model.predict(img_arr)
    probs = preds[0]
    idx = int(np.argmax(probs))
    return idx, probs

# ------------- Debug helper -------------
def print_conv_layers(model):
    print("=== Liste des couches (nom, type, output_shape) ===")
    def rec(layer, prefix=""):
        subs = getattr(layer, "layers", None)
        if subs:
            for sl in subs:
                rec(sl, prefix + layer.name + ".")
        else:
            try:
                shape = getattr(layer, "output", None)
                if shape is not None:
                    print(prefix + layer.name, type(layer).__name__, getattr(layer.output, "shape", None))
                else:
                    print(prefix + layer.name, type(layer).__name__, "no-output")
            except Exception:
                print(prefix + layer.name, type(layer).__name__, "shape-inaccessible")
    rec(model)

# ---------------- robust grad-cam helpers ----------------
def find_last_conv_layer_obj(model):
    """
    Recherche récursive et renvoie l'objet couche (Layer) le plus profond
    dont la sortie est 4D (batch, h, w, channels).
    """
    candidate = None

    def search(lay):
        nonlocal candidate
        subs = getattr(lay, "layers", None)
        if subs:
            for sl in subs:
                search(sl)
        # vérifier output shape
        try:
            out = getattr(lay, "output", None)
            if out is not None:
                shp = None
                try:
                    shp = lay.output.shape
                except Exception:
                    shp = None
                if shp is not None and hasattr(shp, "__len__") and len(shp) == 4:
                    candidate = lay
        except Exception:
            pass

    search(model)
    # fallback: parcourir couches inversées du model simple
    if candidate is None:
        for layer in reversed(getattr(model, "layers", [])):
            try:
                if hasattr(layer, "output") and hasattr(layer.output, "shape") and len(layer.output.shape) == 4:
                    return layer
            except Exception:
                continue
        return None
    return candidate

def make_gradcam_heatmap(model, img_array, class_index, layer_obj=None):
    """
    Retourne heatmap numpy 2D normalisée (valeurs 0..1).
    layer_obj : objet couche (Layer). Si None, on tente find_last_conv_layer_obj.
    """
    if layer_obj is None:
        layer_obj = find_last_conv_layer_obj(model)
        print("Layer object choisi pour Grad-CAM :", getattr(layer_obj, "name", None))
    if layer_obj is None:
        raise ValueError("Impossible de trouver une couche conv utile pour Grad-CAM.")

    # S'assurer que le modèle a été appelé au moins une fois (construit)
    try:
        if not getattr(model, "built", True):
            model(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32))
    except Exception:
        try:
            model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32))
        except Exception:
            pass

    try:
        conv_output = layer_obj.output
        model_output = model.output
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[conv_output, model_output])
    except Exception as e:
        raise RuntimeError(f"Impossible de construire grad_model : {e}")

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Les gradients sont None — la couche choisie peut ne pas être connectée à la sortie.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.reduce_max(heatmap)
    if max_heat.numpy() != 0:
        heatmap /= max_heat

    return heatmap.numpy()

def apply_heatmap_on_image(orig_img_pil, heatmap, alpha=0.5):
    # heatmap is 2D float 0..1
    heatmap_resized = cv2.resize(heatmap, orig_img_pil.size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_bgr = np.array(orig_img_pil)[:, :, ::-1]
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    overlay_rgb = overlay[:, :, ::-1]
    return Image.fromarray(overlay_rgb)

def export_pdf_file(image_path, overlay_pil, pred_class, prob, class_probs, model_name, save_path, class_names):
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Rapport de prédiction - Thermographie PV")
    y -= 22
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 14
    c.drawString(margin, y, f"Modèle: {model_name}")
    y -= 18

    # images côte à côte
    orig = Image.open(image_path).convert("RGB")
    w_img = int((width - 3*margin)/2)
    h_img = min(400, height - 300)
    orig.thumbnail((w_img, h_img))
    overlay_p = overlay_pil.copy()
    overlay_p.thumbnail((w_img, h_img))

    x_orig = margin
    x_overlay = margin + w_img + margin/2
    c.drawImage(ImageReader(orig), x_orig, y-h_img, width=w_img, height=h_img)
    c.drawImage(ImageReader(overlay_p), x_overlay, y-h_img, width=w_img, height=h_img)
    y = y - h_img - 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, f"Prédiction: {pred_class} (probabilité: {prob*100:.2f}%)")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, "Probabilités détaillées:")
    y -= 14
    for i, p in enumerate(class_probs):
        cname = class_names[i] if i < len(class_names) else f"class_{i}"
        c.drawString(margin+8, y, f"- {cname:12s}: {p*100:.2f}%")
        y -= 12

    c.setFont("Helvetica", 8)
    c.drawString(margin, 20, "Généré par PV Thermography inference tool")
    c.save()
    return save_path

# ----- GUI -----
class PVApp:
    def __init__(self, root):
        self.root = root
        root.title("PV Thermography - Inference (improved)")
        root.geometry("1400x820")

        # Grid config
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(0, weight=1)

        # Left: image display with Canvas + scrollbars
        img_container = tk.Frame(root, bd=2, relief="sunken")
        img_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        img_container.grid_rowconfigure(0, weight=1)
        img_container.grid_columnconfigure(0, weight=1)

        # Canvas with scrollbars
        self.canvas = tk.Canvas(img_container, background="#fafafa")
        self.scroll_y = tk.Scrollbar(img_container, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(img_container, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        self.scroll_y.grid(row=0, column=1, sticky="ns")
        self.scroll_x.grid(row=1, column=0, sticky="ew")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Frame inside canvas
        self.img_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.img_frame, anchor="nw")
        self.img_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # padding frame to center
        self.img_container = tk.Frame(self.img_frame, padx=20, pady=20, bg="#fafafa")
        self.img_container.pack(expand=True, fill=tk.BOTH)
        self.img_label = tk.Label(self.img_container, text="Aucune image", bg="#fafafa")
        self.img_label.pack(expand=True)

        # Right: controls + plots
        right = tk.Frame(root)
        right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctrl = tk.Frame(right)
        ctrl.grid(row=0, column=0, sticky="ew", pady=4)

        self.load_img_btn = tk.Button(ctrl, text="Charger image", command=self.on_load_image)
        self.load_img_btn.pack(side=tk.LEFT, padx=4)

        self.predict_btn = tk.Button(ctrl, text="Prédire", command=self.on_predict, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=4)

        self.gc_var = tk.IntVar(value=1)
        self.gc_check = tk.Checkbutton(ctrl, text="Grad‑CAM (overlay)", variable=self.gc_var)
        self.gc_check.pack(side=tk.LEFT, padx=6)

        self.pdf_btn = tk.Button(ctrl, text="Exporter PDF", command=self.on_export_pdf, state=tk.DISABLED)
        self.pdf_btn.pack(side=tk.LEFT, padx=4)

        # Results text (hidden until predict)
        self.result_text = tk.Text(right, width=48, height=10)
        self.result_text.grid(row=1, column=0, pady=8, sticky="nsew")
        self.result_text.insert("1.0", "Aucune prédiction — chargez une image")
        self.result_text.config(state=tk.DISABLED)

        # Histogram (initially hidden)
        fig_frame = tk.Frame(right)
        fig_frame.grid(row=2, column=0, pady=6, sticky="nsew")
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(5.5, 3.5))
        self.ax.set_title("Probabilités par classe")
        # horizontal bar chart placeholder
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=fig_frame)
        # hide initially
        self.canvas_fig.get_tk_widget().pack_forget()

        # placeholders / state
        self.model = None
        self.model_path = None
        self.class_names = try_load_class_names() or DEFAULT_CLASS_NAMES
        self.current_image_path = None
        self.current_orig_pil = None
        self.current_overlay = None
        self.last_pred = None
        self.tk_img = None

        # load model at startup
        try:
            self.model, self.model_path = load_model_auto()
            if self.model is None:
                messagebox.showwarning("Avertissement", "Aucun modèle chargé automatiquement. Tu peux en sélectionner un via la boîte de dialogue.")
            else:
                messagebox.showinfo("Modèle", f"Modèle chargé: {os.path.basename(self.model_path)}")
                # try call once to ensure built
                try:
                    dummy_input = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
                    self.model.predict(dummy_input)
                except Exception as e:
                    print("Note: modèle peut nécessiter un appel lors de la première prédiction:", e)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur au chargement du modèle : {e}")

    def on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event=None):
        # keep window width in sync
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def on_load_image(self):
        path = filedialog.askopenfilename(title="Choisir une image thermique",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.current_image_path = path
        pil, _ = preprocess_image(path)
        self.current_orig_pil = pil
        self.current_overlay = pil.copy()
        self.display_image()
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Image chargée — cliquez sur Prédire")
        self.result_text.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.NORMAL)
        self.pdf_btn.config(state=tk.DISABLED)
        # hide histogram until predict
        self.canvas_fig.get_tk_widget().pack_forget()

    def display_image(self):
        if not self.current_orig_pil:
            return
        display = self.current_orig_pil.copy()
        display.thumbnail(DISPLAY_MAX, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(display)
        self.img_label.configure(image=self.tk_img, text="")
        self.img_label.image = self.tk_img
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_predict(self):
        if self.model is None:
            self.model, self.model_path = load_model_auto()
            if self.model is None:
                messagebox.showerror("Erreur", "Aucun modèle disponible.")
                return
            try:
                dummy_input = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
                self.model.predict(dummy_input)
            except Exception:
                pass

        if not self.current_image_path:
            messagebox.showerror("Erreur", "Aucune image chargée.")
            return

        _, arr = preprocess_image(self.current_image_path)
        try:
            idx, probs = predict(self.model, arr)
        except Exception as e:
            messagebox.showerror("Erreur prédiction", f"Impossible d'effectuer la prédiction : {e}")
            return

        class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        # display results
        text_lines = []
        text_lines.append(f"Fichier: {os.path.basename(self.current_image_path)}")
        text_lines.append(f"Prédiction: {class_name} (prob={probs[idx]:.3f})")
        text_lines.append("\nProbabilités:")
        for i, p in enumerate(probs):
            cn = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            text_lines.append(f" - {cn:12s}: {p:.3f}")

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(text_lines))
        self.result_text.config(state=tk.DISABLED)

        self.last_pred = {"class": class_name, "prob": float(probs[idx]), "probs": [float(x) for x in probs]}

        # update horizontal bar chart (show)
        self.ax.clear()
        names = [self.class_names[i] if i < len(self.class_names) else f"class_{i}" for i in range(len(probs))]
        x_pos = np.arange(len(names))
        self.ax.bar(x_pos, probs, align='center')  # barres verticales
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(names, rotation=30, ha='right')
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probabilité")
        for xi, yi in zip(x_pos, probs):
            self.ax.text(xi, yi + 0.01, f"{yi:.2f}", ha='center', va='bottom')
        self.fig.tight_layout()
        # show canvas if hidden
        widget = self.canvas_fig.get_tk_widget()
        if not widget.winfo_ismapped():
            widget.pack(fill=tk.BOTH, expand=True)
        self.canvas_fig.draw()

        # Grad-CAM overlay (robust)
        if self.gc_var.get():
            try:
                # ensure model built
                try:
                    if not getattr(self.model, "built", True):
                        self.model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32))
                except Exception:
                    pass

                # find conv layer object
                conv_layer_obj = find_last_conv_layer_obj(self.model)
                if conv_layer_obj is None:
                    # show debug info to help
                    print("Aucune couche conv trouvée automatiquement ; imprimez la structure via print_conv_layers.")
                    raise ValueError("Aucune couche convolutionnelle trouvée pour Grad-CAM.")
                # compute heatmap
                heatmap = make_gradcam_heatmap(self.model, arr, idx, layer_obj=conv_layer_obj)
                overlay = apply_heatmap_on_image(self.current_orig_pil, heatmap, alpha=0.5)
                self.current_overlay = overlay
                self.current_orig_pil = overlay.copy()
                self.display_image()
            except Exception as e:
                print("Grad-CAM failed:", e)
                messagebox.showwarning("Grad-CAM", f"Grad-CAM échoué : {e}")
                # fallback: keep original image
                self.current_overlay = self.current_orig_pil
        else:
            self.current_overlay = self.current_orig_pil

        self.pdf_btn.config(state=tk.NORMAL)

    def on_export_pdf(self):
        if not self.last_pred or not self.current_image_path or not self.current_overlay:
            messagebox.showerror("Erreur", "Il manque des données pour exporter le PDF.")
            return

        default_name = f"rapport_{os.path.splitext(os.path.basename(self.current_image_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=default_name, filetypes=[("PDF","*.pdf")])
        if not path:
            return
        if os.path.exists(path):
            ok = messagebox.askyesno("Confirmer", f"{path} existe déjà. Voulez-vous l'écraser ?")
            if not ok:
                return
        try:
            # Générer l'histogramme en image temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig, ax = plt.subplots(figsize=(4, 3))
                names = [self.class_names[i] if i < len(self.class_names) else f"class_{i}" for i in range(len(self.last_pred["probs"]))]
                x_pos = np.arange(len(names))
                ax.bar(x_pos, self.last_pred["probs"], align='center')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(names, rotation=30, ha='right')
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probabilité")
                for xi, yi in zip(x_pos, self.last_pred["probs"]):
                    ax.text(xi, yi + 0.01, f"{yi:.2f}", ha='center', va='bottom')
                fig.tight_layout()
                fig.savefig(tmpfile.name, bbox_inches='tight')
                plt.close(fig)
                hist_path = tmpfile.name

            saved = export_pdf_file(self.current_image_path, self.current_overlay, self.last_pred["class"],
                                    self.last_pred["prob"], self.last_pred["probs"],
                                    os.path.basename(self.model_path) if self.model_path else "model",
                                    path, self.class_names)
            messagebox.showinfo("PDF", f"PDF généré: {saved}")
        except Exception as e:
            messagebox.showerror("Erreur PDF", f"Impossible de générer le PDF : {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PVApp(root)
    root.mainloop()
