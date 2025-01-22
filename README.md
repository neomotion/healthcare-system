# HEALTHCARE SYSTEM
# Healthcare Data Processing & Analysis System

A comprehensive healthcare data processing system that handles both tabular and image data through unified pipelines, featuring model building, training capabilities, and a web interface for easy interaction.

## 🌟 Features

### Data Processing Pipelines
- **Tabular Data Pipeline**
  - Automated data cleaning and preprocessing
  - Feature engineering and selection
  - Missing value handling
  - Categorical variable encoding

- **Image Data Pipeline**
  - Image preprocessing and normalization
  - Augmentation capabilities
  - Batch processing
  - Format standardization
    
![Screenshot from 2025-01-22 18-27-00](https://github.com/user-attachments/assets/ae8953c2-4a88-4772-a00f-e2ede5c46ccc)


- **Unified Pipeline**
  - Seamless handling of both tabular and image data
  - Automated data type detection
  - Parallel processing capabilities
  - Scalable architecture
    
 ![Screenshot from 2025-01-22 18-24-22](https://github.com/user-attachments/assets/36a78f16-c2d9-4ca3-b465-db80dbb35636)


### Model Building & Training
- Support for multiple machine learning algorithms
- Cross-validation and model evaluation
- Hyperparameter optimization
- Model performance metrics and visualization
- Model persistence and versioning

### Web Interface
- User-friendly dashboard
- Data upload and visualization
- Model training monitoring
- Results display and export
  
![Screenshot from 2025-01-22 18-25-04](https://github.com/user-attachments/assets/d0179662-1317-4744-9b75-a2ab9771591a)


## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/healthcare-system.git

# Navigate to project directory
cd healthcare-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Pipelines

```python
from src.pipelines import UnifiedPipeline

# Initialize pipeline
pipeline = UnifiedPipeline()

# Process data
processed_data = pipeline.process(data_path='path/to/data')

# Train model
model = pipeline.train_model(processed_data)
```

### Starting the Web Interface

```bash
# Navigate to web directory
cd src/web

# Run the application
python app.py
```

Visit `http://localhost:5000` in your browser to access the web interface.

## 📋 Requirements

-Run requirements.txt to instal all dependencies


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
