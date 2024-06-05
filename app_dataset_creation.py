import tkinter as tk
from tkinter import filedialog, messagebox
from dataset_creation import create_dataset_llama3  # Assurez-vous d'importer correctement votre fonction

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Créer Dataset pour Llama 3")

        self.input_csv = ""
        self.output_dataset = ""
        
        # Input CSV
        self.input_label = tk.Label(root, text="Fichier CSV d'entrée")
        self.input_label.pack()
        self.input_button = tk.Button(root, text="Charger CSV", command=self.load_input_csv)
        self.input_button.pack()

        # Output Dataset
        self.output_label = tk.Label(root, text="Fichier de sortie du dataset")
        self.output_label.pack()
        self.output_button = tk.Button(root, text="Choisir fichier de sortie", command=self.load_output_dataset)
        self.output_button.pack()

        # ID Starting Estimate Column
        self.id_starting_estimate_label = tk.Label(root, text="Colonne de début de l'estimation d'ID")
        self.id_starting_estimate_label.pack()
        self.id_starting_estimate_entry = tk.Entry(root)
        self.id_starting_estimate_entry.insert(0, "4")
        self.id_starting_estimate_entry.pack()

        # Length Nutrition Table
        self.length_nutrition_label = tk.Label(root, text="Longueur de la table de nutrition")
        self.length_nutrition_label.pack()
        self.length_nutrition_entry = tk.Entry(root)
        self.length_nutrition_entry.insert(0, "40")
        self.length_nutrition_entry.pack()

        # Save to Disk
        self.save_to_disk_var = tk.BooleanVar(value=True)
        self.save_to_disk_check = tk.Checkbutton(root, text="Enregistrer sur le disque", variable=self.save_to_disk_var)
        self.save_to_disk_check.pack()

        # Push to Hub
        self.push_to_hub_var = tk.BooleanVar(value=False)
        self.push_to_hub_check = tk.Checkbutton(root, text="Publier sur le hub", variable=self.push_to_hub_var)
        self.push_to_hub_check.pack()

        # Tokens for Transformer
        self.tokens_label = tk.Label(root, text="Jetons pour le transformer")
        self.tokens_label.pack()
        self.tokens_entry = tk.Entry(root)
        self.tokens_entry.pack()

        # Run Button
        self.run_button = tk.Button(root, text="Créer Dataset", command=self.run_create_dataset)
        self.run_button.pack(pady=20)

    def load_input_csv(self):
        self.input_csv = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Choisir un fichier CSV d'entrée"
        )
        if self.input_csv:
            messagebox.showinfo("Fichier chargé", f"Fichier CSV chargé : {self.input_csv}")

    def load_output_dataset(self):
        self.output_dataset = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Choisir le fichier de sortie du dataset"
        )
        if self.output_dataset:
            messagebox.showinfo("Fichier sélectionné", f"Fichier de sortie : {self.output_dataset}")

    def run_create_dataset(self):
        if not self.input_csv or not self.output_dataset:
            messagebox.showwarning("Attention", "Veuillez charger les fichiers nécessaires.")
            return

        try:
            id_starting_estimate_column = int(self.id_starting_estimate_entry.get())
            length_nutrition_table = int(self.length_nutrition_entry.get())
            save_to_disk = self.save_to_disk_var.get()
            push_to_hub = self.push_to_hub_var.get()
            tokens_for_transformer = self.tokens_entry.get()

            create_dataset_llama3(
                self.input_csv,
                self.output_dataset,
                id_starting_estimate_column,
                length_nutrition_table,
                save_to_disk,
                push_to_hub,
                tokens_for_transformer
            )
            messagebox.showinfo("Succès", "Dataset créé avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
