# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# PKD-for-BERT-Model-Compression Project Guide

## Project Overview
This repository originally implemented **Patient Knowledge Distillation (PKD)** for BERT model compression based on the 2019 paper by Sun et al.

**CURRENT GOAL**: Convert this from research experiment codebase into a clean implementation focused on applying PKD methodology to HuggingFace BERT models. The goal is to remove experiment/benchmark code and create a clean API for PKD with HuggingFace models.

### Original Purpose vs New Direction
- **Original**: Research implementation with GLUE benchmarks and experimental scripts
- **New Goal**: Clean PKD implementation for custom datasets with HuggingFace models
- **Focus**: Apply PKD to legal domain BERT models on legal classification tasks

### Key Features
- Patient Knowledge Distillation (PKD) implementation
- Traditional Knowledge Distillation (KD) support
- GLUE benchmark evaluation
- Multiple distillation loss functions
- Support for various NLP tasks (RTE, MRPC, etc.)

## Technology Stack

### Core Dependencies
- **Python**: Main programming language
- **PyTorch**: Deep learning framework
- **transformers/pytorch-pretrained-bert**: BERT model implementation
- **tqdm**: Progress bars
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **boto3**: AWS S3 integration
- **requests**: HTTP requests for model downloads

### BERT Implementation
The project uses a custom BERT implementation located in the `BERT/` directory, which includes:
- Model architecture definitions
- Tokenization utilities
- Optimization functions
- Example scripts for various NLP tasks

## Project Structure

```
PKD-for-BERT-Model-Compression/
├── BERT/                           # Custom BERT implementation
│   ├── pytorch_pretrained_bert/    # Core BERT modules
│   ├── examples/                   # Example scripts and notebooks
│   └── requirements.txt           # BERT-specific dependencies
├── src/                           # Main source code
│   ├── argument_parser.py         # Command line argument parsing
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── KD_loss.py                 # Knowledge distillation loss functions
│   ├── modeling.py                # Custom model architectures
│   ├── nli_data_processing.py     # NLI task data processing
│   ├── race_data_processing.py    # RACE dataset processing
│   └── utils.py                   # Utility functions
├── scripts/                       # Experimental scripts
│   ├── run_teacher.py             # Teacher model training
│   ├── run_student.py             # Student model training
│   ├── run_student_patience.py    # PKD training
│   └── run_teacher_prediction.py  # Teacher predictions for KD
├── data/                          # Data directory
│   ├── data_raw/                  # Raw datasets (e.g., RTE)
│   ├── data_feat/                 # Tokenized features (optional)
│   ├── models/                    # Pretrained models
│   └── outputs/                   # Training outputs and predictions
├── models/                        # Model storage
│   ├── legal-bert-base/           # Legal BERT base model (12-layer, 768 hidden)
│   └── legal-bert-small/          # Legal BERT small model (6-layer, 512 hidden)
├── NLI_KD_training.py             # Main training script
├── run_glue_benchmark.py          # GLUE benchmark evaluation
├── envs.py                        # Environment configuration
└── requirements.txt               # Python dependencies
```

## Key Components

### Main Training Scripts
- **`NLI_KD_training.py`**: Primary training script with debug mode for quick testing
- **`run_glue_benchmark.py`**: GLUE benchmark evaluation and teacher prediction generation

### Core Modules
- **`src/KD_loss.py`**: Implements distillation and patience loss functions
- **`src/modeling.py`**: Custom BERT model architectures for teacher/student setup
- **`src/argument_parser.py`**: Comprehensive argument parsing with predefined configurations
- **`src/utils.py`**: Model loading, evaluation, and utility functions

### Loss Functions
The project implements three types of losses:
1. **L_CE**: Cross-entropy loss (standard classification loss)
2. **L_DS**: Distillation loss (KL divergence between teacher and student predictions)
3. **L_PT**: Patient loss (MSE between teacher and student intermediate representations)

Combined objective: `L = (1 - α) * L_CE + α * L_DS + β * L_PT`

## Development Workflow

### Environment Setup
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

# Install project dependencies
pip install -r requirements.txt
```

### Data Preparation
1. Modify `HOME_DATA_FOLDER` in `envs.py` (default: `./data`)
2. Create directory structure:
   - `data/data_raw/`: Store raw datasets (MRPC, RTE, etc.)
   - `data/data_feat/`: Store tokenized data (optional)
   - `data/models/pretrained/`: Store pretrained BERT models

### Training Workflow

#### 1. Teacher Training
```python
# In NLI_KD_training.py, set DEBUG=True and uncomment:
argv = get_predefine_argv('glue', 'RTE', 'finetune_teacher')
```

#### 2. Student Training (Standard Fine-tuning)
```python
argv = get_predefine_argv('glue', 'RTE', 'finetune_student')
```

#### 3. Generate Teacher Predictions
Run `run_glue_benchmark.py` with:
- `output_all_layers = True` for Patient Teacher (PKD)
- `output_all_layers = False` for Normal Teacher (KD)

#### 4. Knowledge Distillation
```python
# Vanilla KD
argv = get_predefine_argv('glue', 'RTE', 'kd')

# Patient Knowledge Distillation
argv = get_predefine_argv('glue', 'RTE', 'kd.cls')
```

### Available Tasks
- RTE (Recognizing Textual Entailment) - included in repository
- MRPC (Microsoft Research Paraphrase Corpus)
- Other GLUE benchmark tasks

## Configuration

### Environment Variables (`envs.py`)
- `PROJECT_FOLDER`: Project root directory
- `HOME_DATA_FOLDER`: Data directory path
- `HOME_OUTPUT_FOLDER`: Training outputs directory
- `PREDICTION_FOLDER`: Teacher predictions directory

### Model Configurations
The project supports various model configurations through the argument parser:
- Teacher/Student model sizes
- Learning rates and training parameters
- Distillation parameters (α, β, temperature)
- Task-specific settings

## Evaluation and Results
- GLUE benchmark evaluation support
- Model parameter counting utilities
- Performance visualization scripts
- Accuracy vs. model size analysis

## Development Guidelines

### Code Organization
- Keep data processing logic in `src/` modules
- Use predefined argument configurations for reproducibility
- Store experimental scripts in `scripts/` directory
- Follow the established naming conventions for datasets and models

### Model Storage
- Store pretrained models in `data/models/pretrained/`
- Store fine-tuned models in `models/` or `data/outputs/`
- Large model files (*.bin) are excluded from git via `.gitignore`

### Debugging
- Use `DEBUG = True` in `NLI_KD_training.py` for quick testing
- Predefined argument sets available for common training scenarios
- Logging configured at INFO level for detailed output

## Research Context
This implementation is based on the paper:
```
@article{sun2019patient,
    title={Patient Knowledge Distillation for BERT Model Compression},
    author={Sun, Siqi and Cheng, Yu and Gan, Zhe and Liu, Jingjing},
    journal={arXiv preprint arXiv:1908.09355},
    year={2019}
}
```

The project is part of Microsoft's open-source research contributions and follows Microsoft Open Source Code of Conduct.

## Recent Modifications and Implementation Status

### ✅ Completed Work
1. **LegalProcessor Implementation**:
   - Added `LegalProcessor` class to `src/nli_data_processing.py`
   - Handles 3-column TSV format: index, text, label (True/False → "1"/"0")
   - Registered in `processors` and `output_modes` dictionaries
   - Added legal task support in `compute_metrics()` function
   - Handles UTF-8 encoding with error tolerance

2. **Legal Dataset Integration**:
   - Legal dataset located at `data/data_raw/LEGAL/` (3,637 training examples)
   - TSV format: index, text (legal document), label (True/False)
   - Processor converts True/False to "1"/"0" for classification

3. **Teacher Fine-tuning Setup**:
   - Added `get_legal_teacher_argv()` function in `src/argument_parser.py`
   - Supports both legal-bert-base and legal-bert-small teacher fine-tuning
   - Integrated into `NLI_KD_training.py` debug section
   - Output directories created: `data/outputs/legal/`

4. **Model Configuration**:
   - Updated `models/legal-bert-base/bert_config.json` (12-layer BERT-base config)
   - `models/legal-bert-small/bert_config.json` already configured (6-layer)
   - Both models have proper pytorch_model.bin files and vocab.txt

### 🎯 Current Implementation: Teacher Fine-tuning Stage

The PKD methodology requires this workflow:
1. **Teacher Fine-tuning** (READY) - Fine-tune both legal BERT models on legal dataset
2. **Generate Teacher Predictions** (TODO) - Save teacher outputs for KD
3. **Student Knowledge Distillation** (TODO) - Use PKD to create smaller student models

#### To Run Teacher Fine-tuning:
```python
# Edit NLI_KD_training.py, uncomment one of:
argv = get_legal_teacher_argv('legal-bert-base')    # 12-layer teacher
# OR
argv = get_legal_teacher_argv('legal-bert-small')   # 6-layer teacher

# Then run:
python NLI_KD_training.py
```

#### Key Parameters for Legal Teacher Training:
- Task: `legal` (uses LegalProcessor)
- Alpha: `0.0` (pure fine-tuning, no distillation)
- Batch size: 8 (base) / 16 (small)
- Learning rate: 2e-5
- Epochs: 3
- Max sequence: 512 (good for legal documents)

### 🔄 Next Steps (TODO)
1. **Run teacher fine-tuning** for both models
2. **Generate teacher predictions** using the trained models
3. **Create student models** (smaller than 6-layer) for PKD
4. **Implement HuggingFace integration** for the new goal
5. **Clean up experiment code** and create simple API

### 📊 Legal Dataset Details
- **Location**: `data/data_raw/LEGAL/`
- **Format**: 3-column TSV (index, text, label)
- **Size**: 3,637 training examples
- **Labels**: True/False (binary classification)
- **Task**: Legal document classification

## Notes for Future Claude Code Instances

### Critical Implementation Details
- **LegalProcessor**: Handles True/False → "1"/"0" label conversion automatically
- **UTF-8 Encoding**: Fixed in `_read_tsv()` method for legal dataset compatibility
- **Teacher Training**: Use `alpha=0.0` for pure fine-tuning (no distillation)
- **Model Paths**: Legal models in `models/` directory, training outputs in `data/outputs/legal/`

### Current Workflow State
- **Stage 1**: Teacher fine-tuning setup COMPLETE
- **Stage 2**: Need to run actual training (requires PyTorch environment)
- **Stage 3**: Generate teacher predictions for KD
- **Long-term**: Convert to HuggingFace-based clean API

### Key Files Modified
- `src/nli_data_processing.py`: Added LegalProcessor class
- `src/argument_parser.py`: Added get_legal_teacher_argv() function
- `NLI_KD_training.py`: Added legal teacher training options
- `models/legal-bert-base/bert_config.json`: Updated configuration

This implementation maintains the original PKD methodology while adapting it for custom legal domain datasets.