import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class gui(tk.Tk):
    def __init__(self):
        super().__init__()
        global INPUT 
        self.title("Lung Nodule Classification GUI")
        self.label = tk.Label(self, text="Image Input")
        self.label.pack(side = tk.TOP)
        
        self.input_btn = tk.Button(text="Upload Image", command = lambda:self.upload_file())
        self.input_btn.pack(side = tk.TOP)
        
        self.image_label = tk.Label(self)
        self.image_label.pack(side = tk.TOP)
        
        self.output_btn = tk.Button(text ="Nodule Analysis", command = lambda:self.label_nodule())
        self.output_btn.pack(side = tk.TOP)
        
        self.output_label = tk.Label(self, text="idk")
        self.output_label.pack(side = tk.TOP)

    def upload_file(self):
        global img
        f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png')]
        self.filename = filedialog.askopenfilename(filetypes=f_types)
        img = Image.open(self.filename)
        resized_img = img.resize((200,200))
        img = ImageTk.PhotoImage(resized_img)
        self.label.configure(text="Image Selected: "+self.filename)
        self.image_label.configure(image=img)
    
    def label_nodule(self):
        smth = 123
        

if __name__ == '__main__':
    app = gui()
    app.geometry("500x500")
    app.mainloop()