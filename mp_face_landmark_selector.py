import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

DEFAULT_LANDMARK_INDICES = sorted(list(set([
    0, 1, 2, 4, 5, 6, 7, 10, 13, 14, 17, 21, 30, 33, 37, 39, 40, 46, 48, 52, 53, 54, 55, 58,
    61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 91, 93, 98, 103, 105, 107, 109, 127, 132,
    133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161,
    162, 163, 168, 172, 173, 176, 181, 191, 195, 197, 234, 246, 249, 251, 260, 263, 267,
    269, 270, 276, 278, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310,
    311, 312, 314, 321, 323, 327, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375,
    377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 405,
    409, 415, 454, 466
])))


class FaceLandmarkSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Landmark Selector - Webcam/Image")
        self.root.minsize(900, 700)  # Adjusted min height slightly

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.display_width = 640
        self.display_height = 480
        self.using_webcam = False
        self.webcam_active = False
        self.current_image = None
        self.current_frame = None
        self.cap = None
        self.selected_landmark_indices = []
        self.current_landmarks_coords = []
        self.landmark_radius = 3
        self.landmark_color = "cyan"
        self.selected_landmark_color = "red"
        self.show_tesselation = True
        self.show_contours = True

        self.create_ui()
        self.update_status("Ready - Select a source to begin")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        control_panel = ttk.Frame(main_frame)
        control_panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        source_frame = ttk.LabelFrame(control_panel, text="Input Source", padding=5)
        source_frame.pack(side="left", padx=5, fill="x")
        self.load_image_btn = ttk.Button(source_frame, text="Load Image", command=self.load_image)
        self.load_image_btn.pack(side="left", padx=5)
        self.webcam_btn = ttk.Button(source_frame, text="Start Webcam", command=self.toggle_webcam)
        self.webcam_btn.pack(side="left", padx=5)

        display_frame = ttk.LabelFrame(control_panel, text="Display Options", padding=5)
        display_frame.pack(side="left", padx=5, fill="x")
        self.show_tesselation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Wireframe", variable=self.show_tesselation_var,
                        command=self.update_display).pack(side="left", padx=5)
        self.show_contours_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Contours", variable=self.show_contours_var,
                        command=self.update_display).pack(side="left", padx=5)

        selection_frame = ttk.LabelFrame(control_panel, text="Selection Controls", padding=5)
        selection_frame.pack(side="left", padx=5, fill="x")
        ttk.Button(selection_frame, text="Select Default", command=self.select_default_landmarks).pack(side="left",
                                                                                                       padx=5)
        ttk.Button(selection_frame, text="Clear Selection", command=self.clear_selection).pack(side="left", padx=5)
        ttk.Button(selection_frame, text="Save Selection", command=self.save_selection).pack(side="left", padx=5)

        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(0, weight=3)
        main_frame.rowconfigure(1, weight=1)
        self.canvas = tk.Canvas(canvas_frame, width=self.display_width, height=self.display_height, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(1, weight=1)

        ttk.Label(right_panel, text="Selected Landmarks (Indices):").pack(anchor="w")
        list_frame = ttk.Frame(right_panel)
        list_frame.pack(fill="both", expand=True, pady=(0, 5))
        self.landmark_listbox = tk.Listbox(list_frame, height=10)  # Adjusted height
        self.landmark_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.landmark_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.landmark_listbox.config(yscrollcommand=scrollbar.set)

        # Python list display
        ttk.Label(right_panel, text="Selected Indices (Python List):").pack(anchor="w", pady=(5, 0))
        self.selected_indices_text = tk.Text(right_panel, height=3, width=25, wrap=tk.WORD)
        self.selected_indices_text.pack(fill="x", expand=False, pady=(0, 5))
        self.selected_indices_text.config(state=tk.DISABLED)  # Make it read-only initially

        self.copy_list_btn = ttk.Button(right_panel, text="Copy List to Clipboard",
                                        command=self.copy_landmark_list_to_clipboard)
        self.copy_list_btn.pack(anchor="w", pady=(0, 10), fill="x")

        info_text = ("Click on landmarks to select/deselect them.\n"
                     "Selected landmarks will appear in red.\n")
        ttk.Label(right_panel, text=info_text, justify="left").pack(anchor="w", pady=(5, 0))

        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_image(self):
        if self.webcam_active:
            self.stop_webcam()
        filepath = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ('Image files', '*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif'),  # Added tiff/tif
                ('All files', '*.*')
            ]
        )
        if not filepath: return
        try:
            self.update_status(f"Loading image: {os.path.basename(filepath)}...")
            image = cv2.imread(filepath)
            if image is None: raise ValueError(
                "Failed to open image file. Check if the format is supported by OpenCV and the file is not corrupted.")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            max_dim = 800
            if width > max_dim or height > max_dim:
                scale = min(max_dim / width, max_dim / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            # self.display_width = image_rgb.shape[1] # Canvas size is now dynamic
            # self.display_height = image_rgb.shape[0]
            # self.canvas.config(width=self.display_width, height=self.display_height)
            self.current_image = image_rgb
            self.current_frame = None
            self.using_webcam = False
            self.process_image()
        except Exception as e:
            self.update_status(f"Error loading image: {str(e)}")
            messagebox.showerror("Image Error", str(e))

    def toggle_webcam(self):
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): raise ValueError("Could not open webcam")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Preferred webcam width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Preferred webcam height
            self.webcam_active = True
            self.using_webcam = True
            self.webcam_btn.config(text="Stop Webcam")
            self.update_status("Webcam active")
            self.update_webcam_frame()
        except Exception as e:
            self.update_status(f"Webcam error: {str(e)}")
            messagebox.showerror("Webcam Error", str(e))

    def stop_webcam(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.webcam_active = False
        self.webcam_btn.config(text="Start Webcam")
        self.update_status("Webcam stopped")

    def update_webcam_frame(self):
        if not self.webcam_active: return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = rgb_frame
            self.process_frame()
            self.root.after(15, self.update_webcam_frame)
        else:
            self.update_status("Error: Could not read frame from webcam")
            self.stop_webcam()

    def process_image(self):
        if self.current_image is None: return
        self.current_landmarks_coords = []
        self.update_status("Processing face landmarks...")
        try:
            results = self.face_mesh.process(self.current_image)
            if not results.multi_face_landmarks:
                self.update_status("No face detected in the image")
                self.display_image(self.current_image)  # Display image even if no face
                return
            self.display_image_with_landmarks(self.current_image, results.multi_face_landmarks[0])
            landmark_count = len(results.multi_face_landmarks[0].landmark)
            self.update_status(f"Image loaded - {landmark_count} landmarks detected")
        except Exception as e:
            self.update_status(f"Error processing image: {str(e)}")

    def process_frame(self):
        if self.current_frame is None: return
        self.current_landmarks_coords = []
        try:
            results = self.face_mesh.process(self.current_frame)
            if not results.multi_face_landmarks:
                self.display_image(self.current_frame)  # Display frame even if no face
                self.update_status("No face detected in webcam view")
                return
            self.display_image_with_landmarks(self.current_frame, results.multi_face_landmarks[0])
        except Exception as e:
            self.update_status(f"Error processing frame: {str(e)}")

    def display_image(self, img_rgb_original):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        original_h, original_w = img_rgb_original.shape[:2]

        if original_w == 0 or original_h == 0 or canvas_width <= 1 or canvas_height <= 1:
            # Fallback if canvas size is not yet determined or image is empty
            # Directly use original image if canvas is tiny (e.g. during init)
            pil_img_to_show = Image.fromarray(img_rgb_original)
            self.tk_image = ImageTk.PhotoImage(image=pil_img_to_show)
            self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
            return

        scale = min(canvas_width / original_w, canvas_height / original_h)
        display_w = int(original_w * scale)
        display_h = int(original_h * scale)

        if display_w > 0 and display_h > 0:
            display_img_resized = cv2.resize(img_rgb_original, (display_w, display_h))
        else:  # Should not happen if initial checks pass
            display_img_resized = img_rgb_original

        pil_img = Image.fromarray(display_img_resized)
        self.tk_image = ImageTk.PhotoImage(image=pil_img)
        offset_x = (canvas_width - display_w) // 2
        offset_y = (canvas_height - display_h) // 2
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor=tk.NW)

    def display_image_with_landmarks(self, img_rgb_original, face_landmarks):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        original_h, original_w = img_rgb_original.shape[:2]

        if original_w == 0 or original_h == 0 or canvas_width <= 1 or canvas_height <= 1:
            pil_img_to_show = Image.fromarray(img_rgb_original)  # Basic display
            self.tk_image = ImageTk.PhotoImage(image=pil_img_to_show)
            self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
            return

        scale = min(canvas_width / original_w, canvas_height / original_h)
        display_w = int(original_w * scale)
        display_h = int(original_h * scale)

        if not (display_w > 0 and display_h > 0):  # safety check
            display_w, display_h = original_w, original_h  # use original if scale is bad

        display_img_resized = cv2.resize(img_rgb_original, (display_w, display_h))
        draw_img = display_img_resized.copy()
        offset_x = (canvas_width - display_w) // 2
        offset_y = (canvas_height - display_h) // 2

        if self.show_tesselation_var.get():
            overlay = draw_img.copy()
            for connection in self.mp_face_mesh.FACEMESH_TESSELATION:
                start_idx, end_idx = connection
                if start_idx < len(face_landmarks.landmark) and end_idx < len(face_landmarks.landmark):
                    start_pt = face_landmarks.landmark[start_idx]
                    end_pt = face_landmarks.landmark[end_idx]
                    cv2.line(overlay, (int(start_pt.x * display_w), int(start_pt.y * display_h)),
                             (int(end_pt.x * display_w), int(end_pt.y * display_h)), (0, 255, 255), 1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, draw_img, 1 - alpha, 0, draw_img)

        if self.show_contours_var.get():
            for connection in self.mp_face_mesh.FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                if start_idx < len(face_landmarks.landmark) and end_idx < len(face_landmarks.landmark):
                    start_pt = face_landmarks.landmark[start_idx]
                    end_pt = face_landmarks.landmark[end_idx]
                    cv2.line(draw_img, (int(start_pt.x * display_w), int(start_pt.y * display_h)),
                             (int(end_pt.x * display_w), int(end_pt.y * display_h)), (0, 255, 0), 2)

        pil_img = Image.fromarray(draw_img)
        self.tk_image = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor=tk.NW)

        self.current_landmarks_coords = []
        for idx, landmark in enumerate(face_landmarks.landmark):
            x_on_scaled_img = int(landmark.x * display_w)
            y_on_scaled_img = int(landmark.y * display_h)
            x_on_canvas = x_on_scaled_img + offset_x
            y_on_canvas = y_on_scaled_img + offset_y
            self.current_landmarks_coords.append((x_on_canvas, y_on_canvas, idx))
            color = self.selected_landmark_color if idx in self.selected_landmark_indices else self.landmark_color
            self.canvas.create_oval(
                x_on_canvas - self.landmark_radius, y_on_canvas - self.landmark_radius,
                x_on_canvas + self.landmark_radius, y_on_canvas + self.landmark_radius,
                fill=color, outline=color, tags=f"lm_{idx}"
            )

    def update_display(self):
        if self.using_webcam and self.webcam_active:
            if self.current_frame is not None: self.process_frame()
        elif self.current_image is not None:
            self.process_image()
        elif self.current_frame is not None:  # Last webcam frame after stopping
            self.process_frame()
        else:  # No image or frame, but selection might have changed (e.g. Select Default)
            # If canvas is empty, we can't draw points. But list updates.
            # To reflect point color changes on an empty canvas (if we drew them before face detection)
            # This case might be complex if we want to draw points without an image.
            # For now, landmark drawing is tied to having an image/frame.
            pass

    def on_canvas_click(self, event):
        if not self.current_landmarks_coords: return
        click_x, click_y = event.x, event.y
        closest_idx = None
        min_dist_sq = (self.landmark_radius * 3) ** 2

        for lm_x, lm_y, lm_idx in self.current_landmarks_coords:
            dist_sq = (lm_x - click_x) ** 2 + (lm_y - click_y) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = lm_idx

        if closest_idx is not None:
            if closest_idx in self.selected_landmark_indices:
                self.selected_landmark_indices.remove(closest_idx)
            else:
                self.selected_landmark_indices.append(closest_idx)
            # self.selected_landmark_indices.sort() # Sorting handled in update_selection_display
            self.update_selection_display()  # This will sort and update text area
            self.update_display()

    def update_selection_display(self):
        self.landmark_listbox.delete(0, tk.END)
        # Sort indices for consistent display in both listbox and text area
        self.selected_landmark_indices.sort()
        for idx in self.selected_landmark_indices:
            self.landmark_listbox.insert(tk.END, f"Landmark {idx}")

        list_str = str(self.selected_landmark_indices)
        self.selected_indices_text.config(state=tk.NORMAL)
        self.selected_indices_text.delete(1.0, tk.END)
        self.selected_indices_text.insert(tk.END, list_str)
        self.selected_indices_text.config(state=tk.DISABLED)

        count = len(self.selected_landmark_indices)
        self.update_status(f"Selected {count} landmarks" if count > 0 else "No landmarks selected")

    def select_default_landmarks(self):
        self.selected_landmark_indices = list(set(DEFAULT_LANDMARK_INDICES))  # Already sorted
        self.update_selection_display()
        self.update_status(f"Selected {len(self.selected_landmark_indices)} default landmarks.")
        self.update_display()

    def clear_selection(self):
        self.selected_landmark_indices = []
        self.update_selection_display()
        self.update_display()

    def copy_landmark_list_to_clipboard(self):
        if not self.selected_landmark_indices:
            self.update_status("No landmarks selected to copy.")
            return
        list_str = str(self.selected_landmark_indices)  # Already sorted by update_selection_display
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(list_str)
            self.update_status(f"Copied to clipboard: {list_str}")
            messagebox.showinfo("Clipboard", "Selected landmark list copied to clipboard!")
        except tk.TclError:  # pragma: no cover
            self.update_status("Error: Could not access clipboard.")
            messagebox.showerror("Clipboard Error",
                                 "Could not access the clipboard. Ensure clipboard manager is running or try again.")

    def save_selection(self):
        if not self.selected_landmark_indices:
            messagebox.showinfo("Nothing to Save", "No landmarks are currently selected")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Selected Landmark Indices"
        )
        if not filepath: return
        try:
            with open(filepath, 'w') as f:
                # Save sorted list
                sorted_indices = sorted(self.selected_landmark_indices)
                for idx in sorted_indices:
                    f.write(f"{idx}\n")
                f.write("\n# Comma-separated format:\n")
                f.write(", ".join(map(str, sorted_indices)))
                f.write("\n\n# Python list format:\n")
                f.write(str(sorted_indices))
            self.update_status(f"Saved {len(sorted_indices)} landmarks to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving file: {str(e)}")

    def on_closing(self):
        if self.webcam_active:
            self.stop_webcam()
        if self.face_mesh:
            self.face_mesh.close()
        self.root.destroy()


def main():
    try:
        root = tk.Tk()
        app = FaceLandmarkSelectorApp(root)
        # Let Tkinter determine initial size based on content, then user can resize
        # root.update_idletasks()
        # window_width = root.winfo_reqwidth()
        # window_height = root.winfo_reqheight()
        # Using fixed initial size that accommodates the UI
        window_width = 1000
        window_height = 720

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        print("Application started - running main loop")
        root.mainloop()
        print("Application closed")
    except Exception as e:
        print(f"Error starting application: {str(e)}")


if __name__ == "__main__":
    main()
