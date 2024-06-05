import tkinter as tk
from tkinter import filedialog, messagebox
#from finetuning_model import finetuning

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Finetuning Interface")

        self.json_path = ""

        self.load_button = tk.Button(root, text="Charger Fichier JSON", command=self.load_file)
        self.load_button.pack(pady=20)

        self.run_button = tk.Button(root, text="Run Finetuning", command=self.run_finetuning)
        self.run_button.pack(pady=20)

    def load_file(self):
        self.json_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Choisir un fichier JSON"
        )
        if self.json_path:
            messagebox.showinfo("Fichier chargé", f"Fichier chargé : {self.json_path}")

    def run_finetuning(self):
        if self.json_path:
            try:
                #finetuning(self.json_path)
                print(self.json_path)
                messagebox.showinfo("Succès", "Finetuning terminé avec succès.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")
        else:
            messagebox.showwarning("Attention", "Veuillez charger un fichier JSON d'abord.")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
