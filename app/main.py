import ctypes
import customtkinter
from tkinter import ttk, filedialog
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageTk
import threading

# Hide the console window (Windows only)
if __name__ == "__main__":
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.title("Creative Card")
app.geometry("768x1024")
app.resizable(False, False)  # Disable window resizing

# Splash screen with a progress bar for loading the model
splash = customtkinter.CTkToplevel()
splash.title("Creative Card")
splash.geometry("400x200")
splash_label = customtkinter.CTkLabel(splash, text="Loading model pipeline components, please wait...", font=("Roboto", 14))
splash_label.pack(pady=20)
splash.resizable(False, False)  # Disable window resizing

# Progress bar for loading the model
progress = ttk.Progressbar(splash, mode="indeterminate")
progress.pack(pady=20, padx=20)
progress.start()

def load_model():
    # Load the fine-tuned Stable Diffusion XL model
    fine_tuned_model_dir = r"C:\Users\Jethro\PycharmProjects\OJT Project\Model"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        fine_tuned_model_dir,
        torch_dtype=torch.float16,
        use_safetensors=True  # Automatically handles loading from safetensors files if applicable
    )

    pipe = pipe.to("cuda")  # Use GPU for faster generation
    return pipe

def initialize_model():
    global model_pipeline
    model_pipeline = load_model()
    splash.destroy()  # Close the loading screen once the model is loaded
    app.deiconify()  # Show the main application window
    return

# Load the model in a separate thread to avoid freezing the GUI
thread = threading.Thread(target=initialize_model)
thread.start()

def generate():
    # Display the loading bar inside the image frame
    progress = ttk.Progressbar(image_frame, mode="indeterminate")
    progress.place(relx=0.5, rely=0.5, anchor="center")
    progress.start()

    def generate_image():
        prompt = entry.get()

        # Generate the image
        with torch.no_grad():
            generated_image = model_pipeline(prompt).images[0]

        # Resize the image to 5x7 inches at 300 DPI
        dpi = 300
        target_size_in_inches = (5, 7)
        target_size_in_pixels = (int(target_size_in_inches[0] * dpi), int(target_size_in_inches[1] * dpi))
        upscaled_image = generated_image.resize(target_size_in_pixels, Image.Resampling.LANCZOS)

        # Fit the image to the canvas while maintaining aspect ratio
        frame_width = image_frame.winfo_width()
        frame_height = image_frame.winfo_height()
        image_aspect = upscaled_image.width / upscaled_image.height
        frame_aspect = frame_width / frame_height

        if image_aspect > frame_aspect:
            new_width = frame_width
            new_height = int(frame_width / image_aspect)
        else:
            new_height = frame_height
            new_width = int(frame_height * image_aspect)

        resized_image = upscaled_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        global photo
        photo = ImageTk.PhotoImage(resized_image)

        # Clear the canvas and display the resized image
        canvas.delete("all")
        canvas.create_image(frame_width // 2, frame_height // 2, anchor="center", image=photo)
        canvas.config(scrollregion=canvas.bbox("all"))

        # Show the Save Image button
        save_button.grid(row=2, column=1, padx=10, pady=10)

        # Stop and remove the loading bar after the image is generated
        progress.stop()
        progress.place_forget()


    # Run the image generation in a separate thread
    threading.Thread(target=generate_image).start()

def save_image():
    # Open a file dialog to save the image
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        # Save the current image
        global upscaled_image
        upscaled_image.save(file_path)

frame = customtkinter.CTkFrame(master=app)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Creative Card Cover Image Generator",
                               font=("Roboto", 24))
label.grid(row=0, column=0, columnspan=2, pady=12, padx=10)

entry = customtkinter.CTkEntry(master=frame, height=30, width=768,
                               placeholder_text="Enter Prompt: Ex. A Halloween greeting card with three scary "
                                                "pumpkins in a dark background")
entry.grid(row=1, column=0, columnspan=2, pady=12, padx=10, sticky="ew")

# Generate button with fixed width
button = customtkinter.CTkButton(master=frame, text="Generate",  command=generate, fg_color="#275da3", width=150)
button.grid(row=2, column=0, pady=10, padx=10)

# Save Image button with fixed width (initially hidden)
save_button = customtkinter.CTkButton(master=frame, text="Save Image", text_color="#010010", hover_color="#e9edf0", command=save_image, fg_color="#9fcfe6", width=150)
save_button.grid_forget()  # Hide the button initially

# Frame for displaying the image
image_frame = customtkinter.CTkFrame(master=frame, border_color="#e9edf0", border_width=2)
image_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

# Canvas widget for displaying the image
canvas = customtkinter.CTkCanvas(master=image_frame, bg="#4b5355")
canvas.pack(fill="both", expand=True)

# Configure grid to expand properly
frame.grid_rowconfigure(3, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

app.withdraw()  # Hide the main window while the model is loading
app.mainloop()


