# Manual Spot-Check Q&A Items (Full-Text Grounding)

The following 15 spot-check items are prepared for human verification. Each item specifies the question, the paper title / OpenAlex identifier, the expected page number where the supporting ground truth is located, and a brief description of the target passage.

> **Note for Reviewer:** These items are intended for manual inspection by hand to verify factual grounding and page citation correctness. They have not been auto-marked correct.

---

### Item 1: AutoImpute Architecture
- **Question:** What neural network architecture and loss formulation does AutoImpute use for single-cell RNA-seq imputation?
- **Paper:** *AutoImpute: Autoencoder based imputation of single-cell RNA-seq data* (`W2898718502`)
- **Expected Page:** Page 3
- **Topic/Target Passage:** Details of the sparse autoencoder network, latent layer representation, and Mean Squared Error loss computed over observed non-zero expression entries.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 2: AutoImpute Benchmark Datasets
- **Question:** Which empirical scRNA-seq datasets are used to evaluate AutoImpute against competing methods like scImpute and MAGIC?
- **Paper:** *AutoImpute: Autoencoder based imputation of single-cell RNA-seq data* (`W2898718502`)
- **Expected Page:** Page 4
- **Topic/Target Passage:** Performance comparison table listing datasets including Jurkat-293T, Preimplantation, Kolodziejczyk, Usoskin, and Zeisel.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 3: AutoImpute Baseline Methods
- **Question:** What prior imputation algorithms does the AutoImpute paper compare against in its related work?
- **Paper:** *AutoImpute: Autoencoder based imputation of single-cell RNA-seq data* (`W2898718502`)
- **Expected Page:** Page 2
- **Topic/Target Passage:** Overview of existing imputation techniques including scImpute, drImpute, and MAGIC.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 4: AutoImpute Hyperparameters
- **Question:** What optimization hyperparameters (learning rate, activation function, optimizer) are specified for training the autoencoder?
- **Paper:** *AutoImpute: Autoencoder based imputation of single-cell RNA-seq data* (`W2898718502`)
- **Expected Page:** Page 8
- **Topic/Target Passage:** Methods section detailing learning rate, weight decay, Adam optimizer, and hidden layer dimensions.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 5: Zero-Preserving Imputation Concept
- **Question:** Why does the zero-preserving imputation paper argue that non-zero imputation can distort biological data?
- **Paper:** *Zero-preserving imputation of single-cell RNA-seq data* (`W4205559708`)
- **Expected Page:** Page 1
- **Topic/Target Passage:** Distinction between biological zeros (true absence of expression) versus technical dropouts, and why over-imputing introduces false positive correlations.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 6: Zero-Preserving Mathematical Formulation
- **Question:** What low-rank matrix approximation or thresholding technique does the paper employ to preserve zeros?
- **Paper:** *Zero-preserving imputation of single-cell RNA-seq data* (`W4205559708`)
- **Expected Page:** Page 3
- **Topic/Target Passage:** Singular value thresholding and matrix decomposition preserving zero values below the noise threshold.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 7: Zero-Preserving Clustering Performance
- **Question:** How does zero-preserving imputation affect downstream cell-type clustering accuracy on peripheral blood mononuclear cell (PBMC) data?
- **Paper:** *Zero-preserving imputation of single-cell RNA-seq data* (`W4205559708`)
- **Expected Page:** Page 5
- **Topic/Target Passage:** Clustering metrics (Adjusted Rand Index, Normalized Mutual Information) evaluated across PBMC benchmarks.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 8: scIDPMs Forward Diffusion Formulation
- **Question:** How is the forward diffusion process defined in scIDPMs for perturbing single-cell gene expression counts?
- **Paper:** *scIDPMs: Single-Cell RNA-Seq Imputation Using Diffusion Probabilistic Models* (`W4401247064`)
- **Expected Page:** Page 3
- **Topic/Target Passage:** Mathematical equation for the forward Markov chain adding Gaussian noise with variance schedule $\beta_t$ over $T$ timesteps.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 9: scIDPMs Reverse Denoising Architecture
- **Question:** What neural network backbone is used to parameterize the reverse denoising transition in scIDPMs?
- **Paper:** *scIDPMs: Single-Cell RNA-Seq Imputation Using Diffusion Probabilistic Models* (`W4401247064`)
- **Expected Page:** Page 4
- **Topic/Target Passage:** Denoising score network architecture, time embedding injection, and residual connections.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 10: scIDPMs Quantitative Evaluation Metrics
- **Question:** Which numerical evaluation metrics are reported to quantify imputation error in scIDPMs?
- **Paper:** *scIDPMs: Single-Cell RNA-Seq Imputation Using Diffusion Probabilistic Models* (`W4401247064`)
- **Expected Page:** Page 7
- **Topic/Target Passage:** Reporting of Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Pearson correlation coefficients across simulated and real dropouts.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 11: scIDPMs Downstream Trajectory Inference
- **Question:** How does scIDPMs demonstrate utility in reconstructing cell developmental trajectories or pseudo-time?
- **Paper:** *scIDPMs: Single-Cell RNA-Seq Imputation Using Diffusion Probabilistic Models* (`W4401247064`)
- **Expected Page:** Page 10
- **Topic/Target Passage:** Lineage trajectory reconstruction comparing imputed gene profiles with diffusion pseudotime (DPT) algorithms.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 12: SpotDiff Spatial Conditioning
- **Question:** How does SpotDiff incorporate 2D spatial spot coordinates into its diffusion imputation framework?
- **Paper:** *SpotDiff: Spatial Gene Expression Imputation Diffusion with Single-Cell RNA Sequencing Data Integration* (`W4409347308`)
- **Expected Page:** Page 2
- **Topic/Target Passage:** Formulation of spatial location conditioning vectors concatenated or projected into the latent diffusion space.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 13: SpotDiff Cross-Attention Integration
- **Question:** What mechanism in SpotDiff aligns unpaired scRNA-seq reference profiles with spatial transcriptomics tissue slices?
- **Paper:** *SpotDiff: Spatial Gene Expression Imputation Diffusion with Single-Cell RNA Sequencing Data Integration* (`W4409347308`)
- **Expected Page:** Page 4
- **Topic/Target Passage:** Cross-attention transformer block matching cell-type gene distributions from scRNA-seq to spatial spot features.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 14: Medical HyperRAG Hypergraph Construction
- **Question:** How does Medical HyperRAG construct hyperedges to link multiple entities (symptoms, drugs, diagnoses) across medical texts?
- **Paper:** *Medical HyperRAG: A Hypergraph-Enhanced Retrieval-Augmented Generation Framework for Clinical Question Answering* (`W7155793531`)
- **Expected Page:** Page 3
- **Topic/Target Passage:** Formal definition of clinical hypergraph vertices and multi-relational hyperedges grouping entities extracted from medical corpora.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found

---

### Item 15: Medical HyperRAG Benchmark Datasets
- **Question:** Which clinical QA benchmarks are evaluated in the Medical HyperRAG paper, and what baseline models are compared?
- **Paper:** *Medical HyperRAG: A Hypergraph-Enhanced Retrieval-Augmented Generation Framework for Clinical Question Answering* (`W7155793531`)
- **Expected Page:** Page 10
- **Topic/Target Passage:** Experimental results table showing performance on MedQA-USMLE, PubMedQA, and BioASQ alongside standard dense RAG and Graph RAG baselines.
- **Human Verification:** [ ] Verified Correct [ ] Discrepancy Found
