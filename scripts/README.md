This scripts/ folder is organized roughly as following, and should allow anyone to completely copy replicate the results of the paper.

Firstly, in @load_data.py the data for QM9, ESOL, and HCE are retrieved from online and clustered with **@preprocessing.py** (`clustering` for QM9/ESOL, `clustering_hce` for HCE using AtomPair fingerprints). The older `utils/clustering.py` path is no longer used here. Rows with **HCE `pce_1 == 0`** are dropped before clustering. Use @dataset_qc.py to print example QM9 SMILES that fail sanitization, exceed token limits, or (optionally) have [CLS] embedding-norm outliers on base ChemBERTa—so you can choose which rows to exclude before re-clustering.

Secondly, the data is split in @data_splitting.py. After that, the models are trained in training.py. In TL_chem.py all of the mechanistic interpretability techniques are done for all of the models.

