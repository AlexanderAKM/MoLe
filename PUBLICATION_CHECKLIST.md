# MoLe (Molecular Lens) - Publication Readiness Checklist

This checklist outlines the tasks needed to prepare the repository for publication with a scientific paper.

---

## 📝 Documentation (High Priority)

### Main README
- [ ] **Expand main README.md** with:
  - [ ] Project title and brief description (1-2 paragraphs)
  - [ ] Key features and contributions
  - [ ] Installation instructions (step-by-step)
  - [ ] Quick start example
  - [ ] Usage examples for main functionality
  - [ ] Link to paper (when available)
  - [ ] Directory structure explanation
  - [ ] FAQ section
  - [ ] Acknowledgments and funding sources

### Dependencies & Environment
- [ ] **Create `requirements.txt`** with pinned versions
  - Include: torch, transformers, transformer_lens, pandas, scikit-learn, matplotlib, etc.
- [ ] OR **Create `environment.yml`** for conda users
- [ ] **Add Python version requirement** (e.g., Python 3.8+)
- [ ] Test installation on clean environment

### Citation
- [ ] **Create `CITATION.cff`** file with:
  - Authors
  - Title
  - Year
  - DOI (when available)
  - Repository URL
- [ ] OR provide BibTeX citation in README

### Code Documentation
- [ ] **Add comprehensive docstrings** to all functions in:
  - [ ] `utils/tl_conversion.py`
  - [ ] `utils/tl_validation.py`
  - [ ] `utils/tl_ablation.py`
  - [ ] `utils/tl_regression.py`
  - [ ] `utils/plotting.py`
  - [ ] `utils/clustering.py`
  - [ ] `models/chemberta_regressor.py`
  - [ ] `models/encoder_mlp.py`
  - [ ] `models/simple_mlp.py`
  - [ ] `scripts/training.py`
  - [ ] `scripts/data_splitting.py`
  - [ ] `scripts/load_data.py`
- [ ] **Add module-level docstrings** explaining purpose of each file
- [ ] **Create API documentation** (Sphinx or mkdocs)

### Tutorials & Examples
- [ ] **Convert `TL_chem.py`** to:
  - [ ] Clean example script, OR
  - [ ] Jupyter notebook with explanations
- [ ] **Create tutorial notebooks**:
  - [ ] `01_getting_started.ipynb` - Basic usage
  - [ ] `02_training_models.ipynb` - Training pipeline
  - [ ] `03_regression_lens.ipynb` - Running regression lens analysis
  - [ ] `04_ablation_studies.ipynb` - Running ablation experiments
- [ ] **Add `examples/` directory** with minimal working examples

---

## 🧹 Code Quality & Organization

### Code Cleanup
- [ ] **Remove all `__pycache__` directories**
- [ ] **Update `.gitignore`** to include:
  ```
  __pycache__/
  *.pyc
  *.pyo
  .DS_Store
  .vscode/
  .idea/
  *.ipynb_checkpoints
  ```
- [ ] **Add `__init__.py` files** to:
  - [ ] `utils/`
  - [ ] `models/`
  - [ ] `scripts/`
- [ ] **Review commented-out code** in `TL_chem.py` and clean up

### Code Style
- [ ] **Run code formatter** (black, autopep8, or ruff)
- [ ] **Add type hints** consistently across all modules
- [ ] **Check for PEP 8 compliance** with flake8 or pylint
- [ ] **Remove duplicate or unused code**
- [ ] **Add consistent error handling** with informative messages

### Package Structure
- [ ] **Create `setup.py` or `pyproject.toml`** for pip installation:
  ```bash
  pip install mole-chem
  ```
- [ ] **Define package version** (e.g., 0.1.0)
- [ ] **Specify entry points** if needed

---

## 🔬 Reproducibility (Critical)

### Documentation
- [ ] **Create `REPRODUCIBILITY.md`** with:
  - [ ] Step-by-step instructions to reproduce all results
  - [ ] Hardware requirements (GPU specs, RAM)
  - [ ] Expected runtime for each experiment
  - [ ] Random seed documentation
  - [ ] Known issues and limitations

### Scripts
- [ ] **Create master reproduction script** (`reproduce_all.sh` or `reproduce_all.py`)
- [ ] **Test all scripts run independently**:
  - [ ] `scripts/load_data.py`
  - [ ] `scripts/data_splitting.py`
  - [ ] `scripts/training.py`
  - [ ] Main analysis scripts
- [ ] **Add command-line arguments** to scripts for flexibility
- [ ] **Document script execution order** clearly

### Reproducibility Verification
- [ ] **Document all random seeds** used (currently 19237)
- [ ] **Add seed-setting utility** for reproducibility
- [ ] **Test reproduction on clean environment**
- [ ] **Document software versions** (PyTorch, transformers, etc.)

---

## 📊 Data Management

### Data Documentation
- [ ] **Enhance `clustered_data/README.md`** with:
  - [ ] Detailed data descriptions
  - [ ] Dataset statistics (size, splits, distributions)
  - [ ] Clustering methodology explanation
  - [ ] Data preprocessing steps
- [ ] **Document data licenses** and attribution
- [ ] **Add data sources** with URLs in main README

### Data Availability
- [ ] **Decide on data distribution**:
  - [ ] Include in repository (if small), OR
  - [ ] Upload to Zenodo/Figshare with DOI, OR
  - [ ] Provide download script
- [ ] **Create data download script** if needed (`scripts/download_data.py`)
- [ ] **Add data validation checks**

### Data Processing
- [ ] **Document preprocessing pipeline** clearly
- [ ] **Make data processing scripts accessible**
- [ ] **Add data quality checks**

---

## 🤖 Model Management

### Model Documentation
- [ ] **Enhance `trained_models/README.md`** with:
  - [ ] Model architecture descriptions
  - [ ] Training hyperparameters for each model
  - [ ] Performance metrics summary
  - [ ] File size and loading instructions
- [ ] **Document ChemBERTa base model** (DeepChem/ChemBERTa-77M-MLM)
- [ ] **Add model card** for each trained model

### Model Distribution
- [ ] **Decide on model weight distribution**:
  - [ ] Include in repository (if reasonable size), OR
  - [ ] Upload to Zenodo/HuggingFace with DOI, OR
  - [ ] Provide instructions to reproduce models
- [ ] **Add model loading utilities** with clear API
- [ ] **Create model checksum verification**

### Model Usage
- [ ] **Add simple model inference example**
- [ ] **Document expected inputs/outputs**
- [ ] **Add performance benchmarks**

---

## 🧪 Testing & Validation

### Unit Tests
- [ ] **Create `tests/` directory**
- [ ] **Add unit tests** for:
  - [ ] Data loading functions
  - [ ] Model conversion (HF to TransformerLens)
  - [ ] Regression lens calculations
  - [ ] Ablation utilities
  - [ ] Plotting functions
- [ ] **Use pytest or unittest framework**
- [ ] **Add test data fixtures**

### Integration Tests
- [ ] **Add end-to-end pipeline test**
- [ ] **Test with small sample dataset**
- [ ] **Add continuous integration** (GitHub Actions) - optional

### Validation
- [ ] **Verify model conversion accuracy** (already done in code)
- [ ] **Test on different hardware** (CPU/GPU)
- [ ] **Validate results match paper** figures

---

## 📁 Results Organization

### Results Structure
- [ ] **Clean up results folder**:
  - [ ] Remove redundant folders (qm9 vs qm9_1, hce_500_sample)
  - [ ] Use consistent naming convention
  - [ ] Organize by experiment type
- [ ] **Create `results/README.md`** explaining:
  - [ ] What each subfolder contains
  - [ ] How results were generated
  - [ ] Which results correspond to paper figures

### Figure Quality
- [ ] **Ensure all plots are publication quality**:
  - [ ] High DPI (300+ for PDFs)
  - [ ] Clear labels and legends
  - [ ] Consistent styling
  - [ ] Appropriate font sizes
- [ ] **Add figure generation scripts** for reproducibility
- [ ] **Document plot customization options**

---

## 🏷️ Metadata & Repository Setup

### Repository Files
- [ ] **Review and update LICENSE** if needed
- [ ] **Create `CONTRIBUTING.md`** with:
  - [ ] How to contribute
  - [ ] Code style guidelines
  - [ ] Issue reporting guidelines
  - [ ] Pull request process
- [ ] **Create `CHANGELOG.md`** for version tracking
- [ ] **Add `.gitattributes`** for Git LFS if using large files

### README Enhancements
- [ ] **Add badges** to README:
  - [ ] License badge
  - [ ] Python version badge
  - [ ] Paper link badge (when available)
  - [ ] DOI badge (when available)
- [ ] **Add authors and affiliations**
- [ ] **Add contact information**

### Repository Settings
- [ ] **Add repository description** on GitHub
- [ ] **Add topics/tags** (e.g., chemistry, interpretability, transformers)
- [ ] **Create release** (v0.1.0 or v1.0.0)
- [ ] **Add DOI** through Zenodo integration

---

## 🔍 Final Review

### Code Review
- [ ] **Review all code for:**
  - [ ] Bugs or errors
  - [ ] Inefficiencies
  - [ ] Security issues
  - [ ] Hard-coded paths (should be relative or configurable)
- [ ] **Spell-check** all documentation and comments
- [ ] **Check for TODO comments** and resolve them

### Documentation Review
- [ ] **Proofread all documentation**
- [ ] **Test all code examples** in documentation
- [ ] **Verify all links** work
- [ ] **Check formatting** (markdown rendering)

### External Testing
- [ ] **Have colleague test installation** on fresh environment
- [ ] **Get feedback** on documentation clarity
- [ ] **Test README instructions** step-by-step

### Pre-Publication
- [ ] **Create GitHub release** when paper is accepted
- [ ] **Add paper link** to README
- [ ] **Add preprint link** (arXiv) if available
- [ ] **Announce** on relevant platforms (Twitter, Reddit, etc.)
- [ ] **Archive repository** on Zenodo for permanent DOI

---

## Priority Ranking

### 🔴 Critical (Must-Have)
1. Comprehensive README with installation and usage
2. Requirements/environment file with dependencies
3. Citation file (CITATION.cff)
4. Reproducibility documentation
5. Clean up code (remove __pycache__, organize structure)
6. Model and data availability documentation

### 🟡 Important (Should-Have)
7. Tutorial notebooks or clean examples
8. Docstrings for all public functions
9. Results organization and documentation
10. Unit tests for core functionality
11. setup.py/pyproject.toml for pip installation
12. Code formatting and type hints

### 🟢 Nice-to-Have
13. API documentation (Sphinx)
14. Contributing guidelines
15. Continuous integration
16. Badges and metadata
17. CHANGELOG.md

---

## Notes

- The repository currently has good organization but needs documentation expansion
- Consider whether to rename repository to "MoLe" or keep current name
- Trained models are large - decide on Zenodo vs GitHub LFS vs reproduction instructions
- TL_chem.py is notebook-style - convert to clean script or actual notebook
- Multiple qm9 model folders (train_qm9_05, train_qm9_1, etc.) - document purpose or remove

---

**Estimated Time Investment:**
- Documentation: 2-3 days
- Code cleanup & organization: 1-2 days
- Testing & validation: 1-2 days
- Final review: 1 day
- **Total: ~5-8 days of focused work**

