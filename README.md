# OmniParser v2 GUI - Advanced UI Analysis Tool

![OmniParser v2](https://img.shields.io/badge/OmniParser-v2.0-blue) ![Python](https://img.shields.io/badge/Python-3.13+-yellow) ![CUDA](https://img.shields.io/badge/CUDA-Supported-green)

A powerful, user-friendly GUI application for automated UI element detection, OCR, and semantic analysis. Perfect for analyzing gaming interfaces, application UIs, and any visual interface elements.

## 🎯 What Does OmniParser Do?

OmniParser v2 automatically analyzes screenshots and images to:
- **Detect UI elements** (buttons, icons, text fields, menus)
- **Extract text** using advanced OCR technology
- **Generate semantic descriptions** of visual elements
- **Provide precise bounding boxes** for all detected elements
- **Export structured data** for further analysis

## 🚀 Key Features

### 🖼️ Smart Image Display
- **Stretch to Fill**: Images automatically fill available space
- **Dynamic Zoom Controls**: Zoom in/out, fit to screen, actual size
- **Auto-resize**: Images adjust when you resize the window
- **Scrollable Viewing**: Handle large images with smooth scrolling

### ⚙️ Advanced Parameter Control
- **Box Threshold**: Control detection sensitivity (0.01-0.5)
- **IOU Threshold**: Adjust overlap detection (0.1-0.9)
- **Text Threshold**: Fine-tune OCR accuracy (0.1-1.0)
- **OCR Options**: Toggle PaddleOCR, paragraph mode, local semantics
- **Real-time Updates**: See parameter values change as you adjust sliders

### 🔄 Intelligent Model Management
- **Auto-refresh**: Models refresh automatically when parameters change
- **One-click Processing**: Change settings and process in a single click
- **Manual Refresh**: Backup refresh button for edge cases
- **CUDA Optimization**: Automatic GPU memory management

### 📊 Multiple Output Formats
- **Annotated Image**: Visual representation with numbered bounding boxes
- **Parsed Content**: Raw JSON data with all detected elements
- **Elements Table**: Structured table view with sortable columns
- **CSV Export**: Export data for analysis in Excel, Python, etc.

## 🛠️ Installation

### Prerequisites

**Install Cuda 12.8**

**Download Weights**
https://drive.google.com/file/d/1Otyc6swsZkzNyDHdPvPIXbyCky6QhNkg/view?usp=sharing
```bash
# Install Torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
# Core dependencies
pip install customtkinter
pip install pillow
pip install pandas
pip install ultralytics

# OCR dependencies
pip install paddleocr  # or easyocr
```

### Model Setup
Ensure you have the required model weights in the correct directories:
```
weights/
├── icon_detect/
│   └── model.pt
└── icon_caption_florence/
    └── [florence model files]
```

## 🎮 How to Use

### 1. Launch the Application
```bash
python omniparserv2.gui.py
```

### 2. Load an Image
1. Click **"Select Image"** button
2. Choose your screenshot or UI image
3. Supported formats: PNG, JPG, JPEG, BMP, TIFF, GIF

### 3. Adjust Parameters (Optional)

#### Detection Parameters
- **Box Threshold** (0.05 default): Lower = more sensitive detection
- **IOU Threshold** (0.7 default): Higher = less overlapping detections

#### OCR Parameters  
- **Text Threshold** (0.9 default): Higher = more confident text detection
- **Use PaddleOCR**: Toggle between OCR engines
- **Paragraph Mode**: Group text into paragraphs
- **Use Local Semantics**: Enhanced semantic understanding

### 4. Process the Image
Click **"Process Image"** - the tool will:
1. Auto-refresh models if parameters changed
2. Run OCR on the image
3. Detect UI elements
4. Generate semantic descriptions
5. Display results across three tabs

### 5. Explore Results

#### 📷 Annotated Image Tab
- View your image with numbered bounding boxes
- Use zoom controls: **Zoom In**, **Zoom Out**, **Fit to Screen**, **Actual Size**, **Stretch to Fill**
- Images auto-resize when you resize the window

#### 📄 Parsed Content Tab
- Raw JSON output with all detection data
- Copy/paste data for further analysis
- Full semantic information for each element

#### 📋 Elements Table Tab
- Structured table view of all detected elements
- Columns: ID, Type, Text, Coordinates (X1,Y1,X2,Y2), Confidence
- Click **"Export to CSV"** to save data

## 💡 Pro Tips

### 🎯 Getting Better Results

**For Gaming UIs:**
- Lower Box Threshold (0.02-0.04) for detecting small icons
- Enable Local Semantics for gaming-specific element recognition
- Use higher Text Threshold (0.95) for clean game text

**For Application UIs:**
- Default settings work well for most desktop applications
- Enable Paragraph Mode for dense text areas
- Adjust IOU Threshold if elements overlap heavily

### 🖼️ Image Display Tips

**Workflow Optimization:**
- Use **Stretch to Fill** for maximum screen usage (default)
- **Fit to Screen** maintains aspect ratio for precise viewing
- **Zoom In** for detailed inspection of small elements
- **Actual Size** for pixel-perfect analysis

### ⚡ Performance Optimization

**CUDA/GPU Usage:**
- Models automatically refresh when parameters change
- GPU memory is managed automatically
- Use **"Refresh Models"** manually if you encounter issues
- Close other GPU applications for better performance

## 🔧 Advanced Usage

### Batch Processing Workflow
1. Process first image with desired parameters
2. Note the parameter values that work well
3. Use same parameters for similar images
4. Export results to CSV for batch analysis

### Integration with Other Tools
```python
# Example: Load exported CSV data
import pandas as pd
data = pd.read_csv('exported_elements.csv')

# Filter by element type
buttons = data[data['Type'] == 'button']
text_fields = data[data['Type'] == 'text']
```

### Custom Analysis
- Export parsed content JSON for custom processing
- Use bounding box coordinates for automated testing
- Integrate with game automation frameworks

## 🐛 Troubleshooting

### Common Issues

**"Could not execute a primitive" Error:**
- ✅ **Solution**: Use the "Refresh Models" button manually
- This is automatically handled in newer versions

**Model Loading Errors:**
- Verify model files are in correct `weights/` directories
- Check CUDA installation if using GPU
- Restart application if models fail to load

**Performance Issues:**
- Close other GPU-intensive applications
- Reduce image size for faster processing
- Lower batch_size in processing parameters

**Display Issues:**
- Try different zoom modes if image appears incorrectly
- Resize window to refresh display
- Check image format compatibility

### Memory Management
- Application automatically clears GPU memory
- Large images may require more processing time
- Monitor GPU memory usage for very large batches

## 📁 Output File Structure

### CSV Export Format
```csv
ID,Type,Text,X1,Y1,X2,Y2,Confidence
1,button,Click Here,0.123,0.456,0.234,0.567,0.95
2,text,Welcome Message,0.345,0.678,0.456,0.789,0.88
```

### JSON Structure
```json
{
  "type": "button",
  "text": "Click Here", 
  "bbox": [0.123, 0.456, 0.234, 0.567],
  "confidence": 0.95
}
```

## 🎮 Use Cases

### Gaming Analysis
- Analyze game UI layouts
- Extract button positions for automation
- Study interface design patterns
- Performance testing of UI elements

### Application Testing
- Automated UI testing setup
- Accessibility analysis
- Interface documentation
- Cross-platform UI comparison

### Research & Development
- UI/UX research data collection
- Machine learning dataset creation
- Interface design validation
- User experience optimization

## 🔄 Version History

### v2.0 Features
- ✅ Intelligent auto-refresh system
- ✅ Advanced zoom and display controls
- ✅ Stretch-to-fill by default
- ✅ Auto-resize on window changes
- ✅ Enhanced text detection in tables
- ✅ Improved CUDA memory management
- ✅ One-click processing workflow

## 📞 Support

For issues, feature requests, or questions:
1. Check the troubleshooting section above
2. Verify your model files and dependencies
3. Test with the manual "Refresh Models" button
4. Try restarting the application

---

**Happy Analyzing! 🚀**

*OmniParser v2 - Making UI analysis effortless and powerful.*