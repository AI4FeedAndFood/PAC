import tkinter as tk
from tkinter import filedialog, messagebox
#from predict_estimates import lunch_pipeline  # Assurez-vous d'importer correctement votre fonction

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Predict Estimate")

        self.path_config = ""
        
        # Path Config
        self.config_label = tk.Label(root, text="Fichier de configuration JSON")
        self.config_label.pack()
        self.config_button = tk.Button(root, text="Charger Fichier JSON", command=self.load_config)
        self.config_button.pack()

        # Run Button
        self.run_button = tk.Button(root, text="Lancer Pipeline", command=self.run_pipeline)
        self.run_button.pack(pady=20)

    def load_config(self):
        self.path_config = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Choisir un fichier de configuration JSON"
        )
        if self.path_config:
            messagebox.showinfo("Fichier chargé", f"Fichier JSON chargé : {self.path_config}")

    def run_pipeline(self):
        if not self.path_config:
            messagebox.showwarning("Attention", "Veuillez charger un fichier de configuration JSON.")
            return

        try:
            #lunch_pipeline(self.path_config)
            print(self.path_config)
            messagebox.showinfo("Succès", "Pipeline exécuté avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
