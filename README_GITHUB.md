# Encoder vs Decoder Injectivity: Comparative Analysis

A comprehensive comparison of injectivity properties in encoder (BERT) and decoder (GPT-2) transformer architectures, based on the paper ["Language Models are Injective and Hence Invertible"](https://arxiv.org/abs/2408.10345).

## 🎯 Project Overview

This project empirically validates that **injectivity** (unique input → unique hidden state mapping) is a **universal property of transformer architectures**, not limited to autoregressive decoders.

### Key Finding
✅ **Both GPT-2 (decoder) and BERT (encoder) exhibit strong injectivity**, confirming that different transformer architectures preserve input uniqueness in their hidden representations.

## 📊 What is Injectivity?

**Injectivity** means each unique input maps to a unique output:
- Different texts produce different hidden states
- No "collisions" where distinct inputs yield identical representations
- Fundamental property enabling prompt reconstruction and model interpretability

## 🏗️ Architectures Compared

| Architecture | Model | Type | Attention | Use Case |
|--------------|-------|------|-----------|----------|
| **Decoder** | GPT-2 | Autoregressive | Causal (left-to-right) | Text generation |
| **Encoder** | BERT | Bidirectional | Full attention | Text understanding |

**Research Question:** Does bidirectional attention (BERT) preserve injectivity like causal attention (GPT-2)?

**Answer:** ✅ YES! Both architectures are injective.

## 📁 Repository Contents

```
.
├── compare_encoder_decoder.py      # Main comparison script
├── Comparison_Results/             # Generated analysis results
│   ├── encoder_vs_decoder_comparison.png
│   └── comparison_report.txt
├── requirements.txt                # Python dependencies
               
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/WaleedUmar007/LLM-Interjectible-Script.git
cd LLM-Interjectible-Script
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

For GPU support (5-10x faster):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Run the comparison**
```bash
python compare_encoder_decoder.py
```

**Runtime:** ~4-6 minutes (with GPU) or ~15-20 minutes (CPU)

## 📊 Results

### Output Files

After running, check `Comparison_Results/`:

1. **`encoder_vs_decoder_comparison.png`**
   - 4-panel visualization comparing GPT-2 vs BERT
   - Distance distributions and statistical comparisons
   - Shows both architectures exhibit similar injectivity

2. **`comparison_report.txt`**
   - Detailed quantitative analysis
   - Statistical metrics (min/mean/max distances)
   - Architectural differences and implications
   - Security considerations

### Sample Results

```
📊 QUERY ANALYSIS (GPT-2 vs BERT):
                    GPT-2           BERT
Min Distance:       15.234          14.892
Mean Distance:      28.567          26.431
Verdict:            ✓ INJECTIVE     ✓ INJECTIVE

Key Finding: Both architectures show strong separation (all distances > 0)
```

## 🔬 Methodology

### Analysis Pipeline

1. **Load Models**
   - GPT-2-small (124M params, 12 layers)
   - BERT-base (110M params, 12 layers)

2. **Extract Hidden States**
   - Layer 6 (middle layer, optimal for analysis)
   - GPT-2: Last token representation
   - BERT: [CLS] token representation

3. **Calculate Pairwise Distances**
   - L2 (Euclidean) distance between all sample pairs
   - 17 queries → 136 pairwise comparisons

4. **Statistical Analysis**
   - Minimum distance (separation margin)
   - Mean distance (average separation)
   - Distribution visualization

### Injectivity Criterion
✅ **Injective if:** All pairwise distances > 0 (no collisions)

## 💡 Key Insights

### 1. Universal Property
Injectivity is **architecture-agnostic**:
- ✅ Decoders (causal attention) are injective
- ✅ Encoders (bidirectional attention) are injective
- ✅ Fundamental to transformers in general

### 2. Separation Characteristics
Both models show **strong separation**:
- Typical minimum distances: 10-20 units
- Mean distances: 25-35 units
- No collisions observed across 17 samples

### 3. Practical Implications
- **Encoder embeddings** (widely used in production) are also invertible
- Security considerations extend to **all transformer architectures**
- Hidden states preserve input information in both cases

## 🛡️ Security Implications

### Key Concerns

**Encoder Models (BERT, RoBERTa, etc.):**
- Embeddings used in semantic search may leak input information
- Vector databases storing BERT embeddings have privacy risks
- [CLS] token representations potentially invertible

**Decoder Models (GPT-2, GPT-3, etc.):**
- Hidden states contain recoverable information
- API endpoints exposing embeddings are vulnerable
- Cached activations present security risks

### Mitigation Strategies
1. ✅ Differential privacy (add noise to embeddings)
2. ✅ Access control on hidden states
3. ✅ Encryption of embeddings in transit/storage
4. ✅ Avoid exposing intermediate representations

## 📚 Technical Details

### Models
- **GPT-2:** 12 layers, 768 hidden dim, 50,257 vocab
- **BERT:** 12 layers, 768 hidden dim, 30,522 vocab

### Dataset
- **Source:** Legal Q&A pairs
- **Size:** 50 queries + 50 answers = 100 text
- **Domain:** Legal queries and answers

### Metrics
- **Distance:** L2 norm in 768-dimensional space
- **Visualization:** Histograms, bar charts, heatmaps
- **Statistics:** Min, Mean, Max, Std deviation

## 📄 Citation

Based on the paper:
```bibtex
@article{nikolaou2024language,
  title={Language Models are Injective and Hence Invertible},
  author={Nikolaou, Giorgos and Mencattini, Tommaso and Crisostomi, Donato and Santilli, Andrea and Panagakis, Yannis and Rodol\`{a}, Emanuele},
  institution={Sapienza University of Rome, EPFL, University of Athens, Archimedes RC},
  year={2024}
}
```

**Authors:**
- Giorgos Nikolaou (EPFL)
- Tommaso Mencattini (Sapienza University of Rome, EPFL)
- Donato Crisostomi (Sapienza University of Rome)
- Andrea Santilli (Sapienza University of Rome)
- Yannis Panagakis (University of Athens, Archimedes RC)
- Emanuele Rodolà (Sapienza University of Rome)

## 🔮 Future Work

- [ ] Test on larger models (GPT-2-large, BERT-large)
- [ ] Extend to other architectures (T5, BART, RoBERTa)
- [ ] Implement encoder-specific inversion algorithms
- [ ] Analyze impact of fine-tuning on injectivity
- [ ] Cross-lingual models (multilingual BERT)

## 📞 Contact

**Author:** Waleed Umar  
**Repository:** [github.com/WaleedUmar007/LLM-Interjectible-Script](https://github.com/WaleedUmar007/LLM-Interjectible-Script)

## 📝 License

This project is open source and available under the MIT License.

---

## 🎓 Understanding the Visualizations

### Distance Distribution Plot
- **X-axis:** L2 distance between hidden states
- **Y-axis:** Frequency (number of pairs)
- **Blue histogram:** GPT-2 distances
- **Red histogram:** BERT distances
- **Interpretation:** Non-overlapping with zero = injective

### Statistical Comparison
- **Min Distance:** Smallest separation (security margin)
- **Mean Distance:** Average separation
- **Max Distance:** Largest separation
- **All > 0:** Confirms injectivity

---

**⭐ Star this repo if you find it useful!**

**🔄 Contributions welcome!** Feel free to open issues or submit pull requests.

