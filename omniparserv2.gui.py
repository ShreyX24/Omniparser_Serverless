import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import base64
import io
import threading
import time
import json

# Import your existing modules (keeping your working code intact)
from util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)
import torch
from ultralytics import YOLO

class OmniParserGUI:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")  # "light" or "dark"
        ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
        
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("OmniParser v2 - Advanced UI Analysis")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)
        
        # Initialize variables
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "weights/icon_detect/model.pt"
        self.image_path = None
        self.som_model = None
        self.caption_model_processor = None
        self.processed_image = None
        self.original_image = None
        self.parsed_content_list = []
        self.label_coordinates = []
        
        # Image display variables
        self.zoom_factor = 1.0
        self.current_image = None
        self.max_display_size = 800
        
        # Track if parameters changed since last processing
        self.parameters_changed = False
        self.last_parameters = None
        
        # Initialize models
        self.init_models()
        
        # Create GUI elements
        self.create_widgets()
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
    def init_models(self):
        """Initialize the ML models"""
        try:
            self.som_model = get_yolo_model(self.model_path)
            self.som_model.to(self.device)
            print(f"Model loaded on {self.device}")
            
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2", 
                model_name_or_path="weights/icon_caption_florence", 
                device=self.device
            )
            print("Caption model loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load models: {str(e)}")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main frames
        self.create_control_panel()
        self.create_main_content()
        
    def create_control_panel(self):
        """Create the left control panel"""
        control_frame = ctk.CTkFrame(self.root, width=300)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        control_frame.grid_propagate(False)
        
        # Title
        title_label = ctk.CTkLabel(control_frame, text="OmniParser v2", 
                                 font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(20, 30))
        
        # Image selection
        image_frame = ctk.CTkFrame(control_frame)
        image_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(image_frame, text="Image Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 10))
        
        self.select_image_btn = ctk.CTkButton(image_frame, text="Select Image", 
                                            command=self.select_image)
        self.select_image_btn.pack(pady=(0, 10))
        
        self.image_path_label = ctk.CTkLabel(image_frame, text="No image selected", 
                                           wraplength=250)
        self.image_path_label.pack(pady=(0, 15))
        
        # Parameters frame
        params_frame = ctk.CTkFrame(control_frame)
        params_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(params_frame, text="Parameters", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 10))
        
        # Box Threshold
        ctk.CTkLabel(params_frame, text="Box Threshold:").pack(anchor="w", padx=15)
        self.box_threshold_var = ctk.DoubleVar(value=0.05)
        self.box_threshold_slider = ctk.CTkSlider(params_frame, from_=0.01, to=0.5, 
                                                variable=self.box_threshold_var, number_of_steps=49)
        self.box_threshold_slider.pack(fill="x", padx=15, pady=(5, 5))
        self.box_threshold_label = ctk.CTkLabel(params_frame, text="0.05")
        self.box_threshold_label.pack(anchor="w", padx=15, pady=(0, 10))
        self.box_threshold_slider.configure(command=self.update_box_threshold_label)
        
        # IOU Threshold
        ctk.CTkLabel(params_frame, text="IOU Threshold:").pack(anchor="w", padx=15)
        self.iou_threshold_var = ctk.DoubleVar(value=0.7)
        self.iou_threshold_slider = ctk.CTkSlider(params_frame, from_=0.1, to=0.9, 
                                                variable=self.iou_threshold_var, number_of_steps=80)
        self.iou_threshold_slider.pack(fill="x", padx=15, pady=(5, 5))
        self.iou_threshold_label = ctk.CTkLabel(params_frame, text="0.7")
        self.iou_threshold_label.pack(anchor="w", padx=15, pady=(0, 10))
        self.iou_threshold_slider.configure(command=self.update_iou_threshold_label)
        
        # Text Threshold
        ctk.CTkLabel(params_frame, text="Text Threshold:").pack(anchor="w", padx=15)
        self.text_threshold_var = ctk.DoubleVar(value=0.9)
        self.text_threshold_slider = ctk.CTkSlider(params_frame, from_=0.1, to=1.0, 
                                                 variable=self.text_threshold_var, number_of_steps=90)
        self.text_threshold_slider.pack(fill="x", padx=15, pady=(5, 5))
        self.text_threshold_label = ctk.CTkLabel(params_frame, text="0.9")
        self.text_threshold_label.pack(anchor="w", padx=15, pady=(0, 15))
        self.text_threshold_slider.configure(command=self.update_text_threshold_label)
        
        # OCR options
        ocr_frame = ctk.CTkFrame(control_frame)
        ocr_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(ocr_frame, text="OCR Options", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 10))
        
        self.use_paddleocr_var = ctk.BooleanVar(value=True)
        self.paddleocr_checkbox = ctk.CTkCheckBox(ocr_frame, text="Use PaddleOCR", 
                                                variable=self.use_paddleocr_var,
                                                command=self.on_parameter_change)
        self.paddleocr_checkbox.pack(anchor="w", padx=15, pady=5)
        
        self.paragraph_mode_var = ctk.BooleanVar(value=False)
        self.paragraph_checkbox = ctk.CTkCheckBox(ocr_frame, text="Paragraph Mode", 
                                                variable=self.paragraph_mode_var,
                                                command=self.on_parameter_change)
        self.paragraph_checkbox.pack(anchor="w", padx=15, pady=5)
        
        self.use_local_semantics_var = ctk.BooleanVar(value=True)
        self.local_semantics_checkbox = ctk.CTkCheckBox(ocr_frame, text="Use Local Semantics", 
                                                      variable=self.use_local_semantics_var,
                                                      command=self.on_parameter_change)
        self.local_semantics_checkbox.pack(anchor="w", padx=15, pady=(5, 15))
        
        # Process button
        self.process_btn = ctk.CTkButton(control_frame, text="Process Image", 
                                       command=self.process_image, height=40,
                                       font=ctk.CTkFont(size=16, weight="bold"))
        self.process_btn.pack(fill="x", padx=20, pady=(0, 10))
        
        # Refresh models button
        self.refresh_btn = ctk.CTkButton(control_frame, text="Refresh Models", 
                                       command=self.manual_refresh_models, height=30,
                                       font=ctk.CTkFont(size=12))
        self.refresh_btn.pack(fill="x", padx=20, pady=(0, 20))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(control_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 20))
        self.progress_bar.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(control_frame, text="Ready", 
                                       font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=(0, 20))
        
    def create_main_content(self):
        """Create the main content area with tabs"""
        # Main content frame
        content_frame = ctk.CTkFrame(self.root)
        content_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Create tabview
        self.tabview = ctk.CTkTabview(content_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Add tabs
        self.tab_image = self.tabview.add("Annotated Image")
        self.tab_content = self.tabview.add("Parsed Content")
        self.tab_elements = self.tabview.add("Elements Table")
        
        self.setup_image_tab()
        self.setup_content_tab()
        self.setup_elements_tab()
        
    def setup_image_tab(self):
        """Setup the annotated image tab with zoom controls"""
        # Create main frame for image tab
        main_frame = ctk.CTkFrame(self.tab_image)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create control buttons frame
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(fill="x", padx=5, pady=(5, 10))
        
        # Zoom controls
        zoom_label = ctk.CTkLabel(controls_frame, text="Image Controls:", font=ctk.CTkFont(weight="bold"))
        zoom_label.pack(side="left", padx=(10, 20))
        
        self.zoom_in_btn = ctk.CTkButton(controls_frame, text="Zoom In", width=80,
                                        command=self.zoom_in)
        self.zoom_in_btn.pack(side="left", padx=5)
        
        self.zoom_out_btn = ctk.CTkButton(controls_frame, text="Zoom Out", width=80,
                                         command=self.zoom_out)
        self.zoom_out_btn.pack(side="left", padx=5)
        
        self.fit_screen_btn = ctk.CTkButton(controls_frame, text="Fit to Screen", width=100,
                                           command=self.fit_to_screen)
        self.fit_screen_btn.pack(side="left", padx=5)
        
        self.actual_size_btn = ctk.CTkButton(controls_frame, text="Actual Size", width=100,
                                            command=self.actual_size)
        self.actual_size_btn.pack(side="left", padx=5)
        
        self.stretch_btn = ctk.CTkButton(controls_frame, text="Stretch to Fill", width=120,
                                        command=self.stretch_to_fill)
        self.stretch_btn.pack(side="left", padx=5)
        
        # Zoom level display
        self.zoom_level_label = ctk.CTkLabel(controls_frame, text="Zoom: 100%")
        self.zoom_level_label.pack(side="right", padx=(20, 10))
        
        # Create scrollable frame for image
        self.image_scroll_frame = ctk.CTkScrollableFrame(main_frame)
        self.image_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.image_label = ctk.CTkLabel(self.image_scroll_frame, text="No image processed yet")
        self.image_label.pack(expand=True)
        
    def setup_content_tab(self):
        """Setup the parsed content tab"""
        self.content_text = ctk.CTkTextbox(self.tab_content)
        self.content_text.pack(fill="both", expand=True, padx=10, pady=10)
        
    def setup_elements_tab(self):
        """Setup the elements table tab"""
        # Create frame for table
        table_frame = ctk.CTkFrame(self.tab_elements)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create treeview for table
        columns = ('ID', 'Type', 'Text', 'X1', 'Y1', 'X2', 'Y2', 'Confidence')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configure column headings and widths
        column_widths = {'ID': 50, 'Type': 100, 'Text': 300, 'X1': 60, 'Y1': 60, 'X2': 60, 'Y2': 60, 'Confidence': 80}
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor='center')
        
        # Add scrollbars
        tree_scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        # Pack elements
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll_y.pack(side="right", fill="y")
        tree_scroll_x.pack(side="bottom", fill="x")
        
        # Export button
        export_btn = ctk.CTkButton(self.tab_elements, text="Export to CSV", 
                                 command=self.export_to_csv)
        export_btn.pack(pady=10)
        
    def zoom_in(self):
        """Zoom in the image"""
        self.zoom_factor *= 1.25
        self.update_image_display()
        
    def zoom_out(self):
        """Zoom out the image"""
        self.zoom_factor /= 1.25
        if self.zoom_factor < 0.1:
            self.zoom_factor = 0.1
        self.update_image_display()
        
    def fit_to_screen(self):
        """Fit image to available screen space"""
        if self.original_image:
            # Force update to get actual frame dimensions
            self.root.update_idletasks()
            
            # Get actual scrollable frame dimensions
            try:
                frame_width = self.image_scroll_frame.winfo_width()
                frame_height = self.image_scroll_frame.winfo_height()
                
                # If frame hasn't been rendered yet, use reasonable defaults
                if frame_width <= 1 or frame_height <= 1:
                    frame_width = 1000
                    frame_height = 700
                
                # Account for padding and scrollbars
                available_width = max(frame_width - 40, 400)
                available_height = max(frame_height - 40, 300)
                
            except:
                # Fallback to larger defaults if there's an error
                available_width = 1000
                available_height = 700
            
            img_width, img_height = self.original_image.size
            
            # Calculate zoom factor to fit both dimensions
            zoom_x = available_width / img_width
            zoom_y = available_height / img_height
            self.zoom_factor = min(zoom_x, zoom_y)  # Allow zooming beyond 100% if needed
            
            # Ensure minimum zoom
            if self.zoom_factor < 0.1:
                self.zoom_factor = 0.1
            
            self.update_image_display()
            
    def actual_size(self):
        """Show image at actual size"""
        self.zoom_factor = 1.0
        self.update_image_display()
        
    def stretch_to_fill(self):
        """Stretch image to fill available space (ignores aspect ratio)"""
        if self.original_image:
            # Force update to get actual frame dimensions
            self.root.update_idletasks()
            
            # Get actual scrollable frame dimensions
            try:
                frame_width = self.image_scroll_frame.winfo_width()
                frame_height = self.image_scroll_frame.winfo_height()
                
                # If frame hasn't been rendered yet, use reasonable defaults
                if frame_width <= 1 or frame_height <= 1:
                    frame_width = 1000
                    frame_height = 700
                
                # Account for padding and scrollbars
                available_width = max(frame_width - 40, 400)
                available_height = max(frame_height - 40, 300)
                
            except:
                # Fallback to larger defaults if there's an error
                available_width = 1000
                available_height = 700
            
            img_width, img_height = self.original_image.size
            
            # Calculate zoom factor to fill space (use the larger ratio)
            zoom_x = available_width / img_width
            zoom_y = available_height / img_height
            self.zoom_factor = max(zoom_x, zoom_y)  # Use max to fill space
            
            # Ensure minimum zoom
            if self.zoom_factor < 0.1:
                self.zoom_factor = 0.1
            
            self.update_image_display()
        
    def update_image_display(self):
        """Update the image display with current zoom factor"""
        if self.original_image:
            # Calculate new size
            width = int(self.original_image.size[0] * self.zoom_factor)
            height = int(self.original_image.size[1] * self.zoom_factor)
            
            # Resize image
            resized_image = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(resized_image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Update zoom level display
            self.zoom_level_label.configure(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        
    def manual_refresh_models(self):
        """Manually refresh models when button is clicked"""
        try:
            self.refresh_btn.configure(state="disabled", text="Refreshing...")
            self.status_label.configure(text="Refreshing models...")
            
            self.refresh_models()
            
            self.status_label.configure(text="Models refreshed successfully")
            messagebox.showinfo("Success", "Models refreshed successfully!")
            
        except Exception as e:
            self.status_label.configure(text="Model refresh failed")
            messagebox.showerror("Error", f"Failed to refresh models: {str(e)}")
        finally:
            self.refresh_btn.configure(state="normal", text="Refresh Models")
    
    def refresh_models(self):
        """Refresh models to clear any state issues - EXACT COPY from working version"""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Reload SOM model
            self.som_model = get_yolo_model(self.model_path)
            self.som_model.to(self.device)
            
            # Caption model usually doesn't need reloading, but clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("Models refreshed successfully")
        except Exception as e:
            print(f"Model refresh error: {e}")
            raise e
    
    def update_box_threshold_label(self, value):
        """Update box threshold label"""
        self.box_threshold_label.configure(text=f"{value:.3f}")
        self.parameters_changed = True
        
    def update_iou_threshold_label(self, value):
        """Update IOU threshold label"""
        self.iou_threshold_label.configure(text=f"{value:.3f}")
        self.parameters_changed = True
        
    def update_text_threshold_label(self, value):
        """Update text threshold label"""
        self.text_threshold_label.configure(text=f"{value:.3f}")
        self.parameters_changed = True
        
    def on_parameter_change(self):
        """Called when any parameter changes"""
        self.parameters_changed = True
        
    def on_window_resize(self, event):
        """Called when window is resized"""
        # Only handle resize for the main window, not child widgets
        if event.widget == self.root and self.original_image:
            # Small delay to avoid too many rapid calls during resize
            self.root.after(200, self.stretch_to_fill)
            
    def get_current_parameters(self):
        """Get current parameter values as a tuple for comparison"""
        return (
            self.box_threshold_var.get(),
            self.iou_threshold_var.get(),
            self.text_threshold_var.get(),
            self.use_paddleocr_var.get(),
            self.paragraph_mode_var.get(),
            self.use_local_semantics_var.get()
        )
        
    def select_image(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.image_path_label.configure(text=f"...{file_path[-30:]}")
            self.status_label.configure(text="Image selected")
            
    def process_image(self):
        """Process the selected image in a separate thread"""
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        if not self.som_model or not self.caption_model_processor:
            messagebox.showerror("Error", "Models not loaded properly!")
            return
            
        # Disable process button and start processing
        self.process_btn.configure(state="disabled", text="Processing...")
        self.progress_bar.set(0)
        self.status_label.configure(text="Processing...")
        
        # Start processing in separate thread
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
        
    def _process_image_thread(self):
        """Process image in separate thread with automatic model refresh when needed"""
        try:
            # Check if parameters have changed and auto-refresh if needed
            current_params = self.get_current_parameters()
            needs_refresh = (self.parameters_changed or 
                           self.last_parameters is None or 
                           current_params != self.last_parameters)
            
            if needs_refresh:
                self.root.after(0, lambda: self.status_label.configure(text="Auto-refreshing models for parameter changes..."))
                # Use the EXACT same refresh method that works manually
                self.refresh_models()
                self.parameters_changed = False
                self.last_parameters = current_params
            
            # Clear CUDA cache and reset model state (like in working version)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Update progress
            self.root.after(0, lambda: self.progress_bar.set(0.2))
            self.root.after(0, lambda: self.status_label.configure(text="Running OCR..."))
            
            # Get image info for scaling
            image = Image.open(self.image_path)
            box_overlay_ratio = max(image.size) / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }
            
            # OCR processing (exactly like working version)
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                self.image_path, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={
                    'paragraph': self.paragraph_mode_var.get(), 
                    'text_threshold': self.text_threshold_var.get()
                }, 
                use_paddleocr=self.use_paddleocr_var.get()
            )
            text, ocr_bbox = ocr_bbox_rslt
            
            self.root.after(0, lambda: self.progress_bar.set(0.6))
            self.root.after(0, lambda: self.status_label.configure(text="Generating annotations..."))
            
            # Main processing with model state refresh (exactly like working version)
            with torch.no_grad():  # Ensure no gradients are computed
                dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                    self.image_path, 
                    self.som_model, 
                    BOX_TRESHOLD=self.box_threshold_var.get(), 
                    output_coord_in_ratio=True, 
                    ocr_bbox=ocr_bbox,
                    draw_bbox_config=draw_bbox_config, 
                    caption_model_processor=self.caption_model_processor, 
                    ocr_text=text,
                    use_local_semantics=self.use_local_semantics_var.get(), 
                    iou_threshold=self.iou_threshold_var.get(), 
                    scale_img=False, 
                    batch_size=128
                )
            
            # Store results
            self.processed_image = dino_labled_img
            self.parsed_content_list = parsed_content_list
            self.label_coordinates = label_coordinates
            
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            self.root.after(0, lambda: self.status_label.configure(text="Updating display..."))
            
            # Update GUI with results
            self.root.after(0, self.update_results)
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            error_msg = str(e)
            print(f"Processing error: {error_msg}")  # For debugging
            
            # Use the EXACT same error handling as the working version
            if "primitive" in error_msg.lower() or "cuda" in error_msg.lower():
                self.root.after(0, lambda: self.status_label.configure(text="Refreshing models..."))
                try:
                    self.refresh_models()
                    self.root.after(0, lambda: self.status_label.configure(text="Models refreshed. Try again."))
                except:
                    self.root.after(0, lambda: messagebox.showerror("Model Error", 
                        "Model refresh failed. Please restart the application."))
            else:
                self.root.after(0, lambda: messagebox.showerror("Processing Error", f"Error processing image: {error_msg}"))
                self.root.after(0, lambda: self.status_label.configure(text="Error occurred"))
        finally:
            self.root.after(0, lambda: self.process_btn.configure(state="normal", text="Process Image"))
            
    def update_results(self):
        """Update GUI with processing results"""
        try:
            # Update annotated image
            if self.processed_image:
                image_data = base64.b64decode(self.processed_image)
                self.original_image = Image.open(io.BytesIO(image_data))
                self.current_image = self.original_image.copy()
                
                # Reset zoom and display image with stretch to fill by default
                self.zoom_factor = 1.0
                # Use after_idle to ensure the frame is properly rendered before stretching
                self.root.after_idle(self.stretch_to_fill)
                
            # Update parsed content
            if self.parsed_content_list:
                content_text = json.dumps(self.parsed_content_list, indent=2)
                self.content_text.delete("1.0", tk.END)
                self.content_text.insert("1.0", content_text)
                
                # Update elements table
                self.update_elements_table()
                
            self.status_label.configure(text=f"Processed successfully! Found {len(self.parsed_content_list)} elements")
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Error updating display: {str(e)}")
            
    def update_elements_table(self):
        """Update the elements table with parsed content"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add new items
        for i, element in enumerate(self.parsed_content_list):
            # Extract relevant information with better text handling
            element_id = i + 1
            element_type = element.get('type', 'Unknown')
            
            # Try multiple possible keys for text content
            text_content = ""
            possible_text_keys = ['text', 'label', 'caption', 'description', 'content', 'value']
            for key in possible_text_keys:
                if key in element and element[key]:
                    text_content = str(element[key])
                    break
            
            # If still no text, check if element itself is a string or has nested text
            if not text_content:
                if isinstance(element, str):
                    text_content = element
                elif 'attributes' in element and isinstance(element['attributes'], dict):
                    for key in possible_text_keys:
                        if key in element['attributes'] and element['attributes'][key]:
                            text_content = str(element['attributes'][key])
                            break
            
            # Truncate long text for display
            display_text = text_content[:100] + '...' if len(text_content) > 100 else text_content
            
            # Get bounding box coordinates
            bbox = element.get('bbox', element.get('coordinates', element.get('box', [0, 0, 0, 0])))
            if isinstance(bbox, dict):
                x1 = bbox.get('x1', bbox.get('left', 0))
                y1 = bbox.get('y1', bbox.get('top', 0))
                x2 = bbox.get('x2', bbox.get('right', 0))
                y2 = bbox.get('y2', bbox.get('bottom', 0))
            else:
                x1, y1, x2, y2 = bbox[:4] if len(bbox) >= 4 else [0, 0, 0, 0]
            
            confidence = element.get('confidence', element.get('score', 0))
            
            self.tree.insert('', 'end', values=(
                element_id, element_type, display_text, 
                f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}", 
                f"{confidence:.3f}"
            ))
            
    def export_to_csv(self):
        """Export elements table to CSV"""
        if not self.parsed_content_list:
            messagebox.showwarning("Warning", "No data to export!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save CSV File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.parsed_content_list)
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")
                
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = OmniParserGUI()
    app.run()