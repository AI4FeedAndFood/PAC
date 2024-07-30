import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import io
import glob
import os 

class PDFCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Cropper Eurofins")
        
        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.zoom_in_button = tk.Button(self.toolbar, text="Zoomer +", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.zoom_out_button = tk.Button(self.toolbar, text="Dézoomer -", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.prev_page_button = tk.Button(self.toolbar, text="Page Précédente", command=self.prev_page)
        self.prev_page_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.next_page_button = tk.Button(self.toolbar, text="Page Suivante", command=self.next_page)
        self.next_page_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.hbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.hbar.config(command=self.canvas.xview)
        self.vbar.config(command=self.canvas.yview)
        
        self.menubar = tk.Menu(root)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Charger PDF", command=self.load_pdf)
        self.menubar.add_cascade(label="Fichier", menu=self.file_menu)
        root.config(menu=self.menubar)
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.pdf_document = None
        self.current_page_number = 0
        self.tk_image = None
        self.scale_x = 2
        self.scale_y = 2
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # For Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # For Linux
        
        # Enable drag and drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

    def load_pdf(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
            if not file_path:
                return
        
        self.pdf_document = fitz.open(file_path)
        self.current_page_number = 0  # Start with the first page
        self.render_page(self.current_page_number)  # Render the first page

    def render_page(self, page_number, zoom=None):
        if zoom:
            self.scale_x = zoom
            self.scale_y = zoom
        self.current_page = self.pdf_document[page_number]
        mat = fitz.Matrix(self.scale_x, self.scale_y)
        pix = self.current_page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes()))
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if not self.current_page:
            return
        
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        x1 = min(self.start_x, end_x) / self.scale_x
        y1 = min(self.start_y, end_y) / self.scale_y
        x2 = max(self.start_x, end_x) / self.scale_x
        y2 = max(self.start_y, end_y) / self.scale_y
        
        # Use a higher zoom factor for better quality
        zoom_factor = 4
        mat = fitz.Matrix(zoom_factor, zoom_factor)
        cropped_image = self.current_page.get_pixmap(matrix=mat, clip=fitz.Rect(x1, y1, x2, y2))
        
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            cropped_image.save(save_path)
            messagebox.showinfo("Succès", f"Image enregistrée sous {save_path}")

    def on_mouse_wheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom_in()
        elif event.num == 5 or event.delta < 0:
            self.zoom_out()

    def zoom_in(self):
        zoom = min(self.scale_x + 0.1, 4)  # Zoom in, max zoom factor 4
        self.render_page(self.current_page_number, zoom)

    def zoom_out(self):
        zoom = max(self.scale_x - 0.1, 0.5)  # Zoom out, min zoom factor 0.5
        self.render_page(self.current_page_number, zoom)
    
    def on_drop(self, event):
        file_path = event.data
        if file_path.endswith('.pdf'):
            self.load_pdf(file_path)
    
    def prev_page(self):
        if self.pdf_document and self.current_page_number > 0:
            self.current_page_number -= 1
            self.render_page(self.current_page_number)
    
    def next_page(self):
        if self.pdf_document and self.current_page_number < len(self.pdf_document) - 1:
            self.current_page_number += 1
            self.render_page(self.current_page_number)


class PDFsCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Cropper Eurofins")
        
        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.zoom_in_button = tk.Button(self.toolbar, text="Zoomer +", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.zoom_out_button = tk.Button(self.toolbar, text="Dézoomer -", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.prev_page_button = tk.Button(self.toolbar, text="Page Précédente", command=self.prev_page)
        self.prev_page_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.next_page_button = tk.Button(self.toolbar, text="Page Suivante", command=self.next_page)
        self.next_page_button.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.hbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.hbar.config(command=self.canvas.xview)
        self.vbar.config(command=self.canvas.yview)
        
        self.menubar = tk.Menu(root)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Charger PDF", command=self.load_pdf_directory)
        self.menubar.add_cascade(label="Fichier", menu=self.file_menu)
        root.config(menu=self.menubar)
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.pdf_document = None
        self.current_page_number = 0
        self.tk_image = None
        self.scale_x = 2
        self.scale_y = 2
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # For Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # For Linux
        
        # Enable drag and drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        self.save_directory = filedialog.askdirectory() + "/"
        self.current_path = ""
        
    def load_pdf(self, file_path=None):
        file_path
        if not file_path:
            file_path = self.pdf_directory + self.pdf_files[self.current_pdf]
            self.current_pdf += 1 
            if not file_path:
                return
        
        self.pdf_document = fitz.open(file_path)
        self.current_page_number = 0  # Start with the first page
        self.render_page(self.current_page_number)  # Render the first page
    
    def get_pdf_files(self):
        pdf_files = glob.glob(os.path.join(self.pdf_directory, '*.pdf'))
        pdf_files = [os.path.basename(f) for f in pdf_files]  # Pour ne garder que les noms de fichiers
        self.pdf_files = pdf_files
        print(pdf_files)
        
    
    def load_pdf_directory(self, file_path=None):
        if not file_path:
            self.pdf_directory = filedialog.askdirectory() + "/"
            
            if not self.pdf_directory:
                return
        self.current_pdf = 0 
        self.get_pdf_files()
        
        self.load_pdf()
        

    def render_page(self, page_number, zoom=None):
        if zoom:
            self.scale_x = zoom
            self.scale_y = zoom
        self.current_page = self.pdf_document[page_number]
        mat = fitz.Matrix(self.scale_x, self.scale_y)
        pix = self.current_page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes()))
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if not self.current_page:
            return
        
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        x1 = min(self.start_x, end_x) / self.scale_x
        y1 = min(self.start_y, end_y) / self.scale_y
        x2 = max(self.start_x, end_x) / self.scale_x
        y2 = max(self.start_y, end_y) / self.scale_y
        
        # Use a higher zoom factor for better quality
        zoom_factor = 4
        mat = fitz.Matrix(zoom_factor, zoom_factor)
        cropped_image = self.current_page.get_pixmap(matrix=mat, clip=fitz.Rect(x1, y1, x2, y2))
        
        save_path = self.save_directory + self.pdf_files[self.current_pdf - 1]
        print(save_path)
        save_path = save_path[:-3] +'png'
        print(save_path)
        save_or_redo = messagebox.askyesno("Confirmation", "Voulez-vous enregistrer l'image ou refaire ? (Oui pour enregistrer, Non pour refaire)")
        if save_or_redo:
            cropped_image.save(save_path)
            messagebox.showinfo("Succès", f"Image enregistrée sous {save_path}")
            self.load_pdf()

    def on_mouse_wheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom_in()
        elif event.num == 5 or event.delta < 0:
            self.zoom_out()

    def zoom_in(self):
        zoom = min(self.scale_x + 0.1, 4)  # Zoom in, max zoom factor 4
        self.render_page(self.current_page_number, zoom)

    def zoom_out(self):
        zoom = max(self.scale_x - 0.1, 0.5)  # Zoom out, min zoom factor 0.5
        self.render_page(self.current_page_number, zoom)
    
    def on_drop(self, event):
        file_path = event.data
        if file_path.endswith('.pdf'):
            self.load_pdf(file_path)
    
    def prev_page(self):
        if self.pdf_document and self.current_page_number > 0:
            self.current_page_number -= 1
            self.render_page(self.current_page_number)
    
    def next_page(self):
        if self.pdf_document and self.current_page_number < len(self.pdf_document) - 1:
            self.current_page_number += 1
            self.render_page(self.current_page_number)



if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Use TkinterDnD for drag and drop
    app = PDFsCropper(root)
    root.geometry("800x600")
    root.mainloop()
