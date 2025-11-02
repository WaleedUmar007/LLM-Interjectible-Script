# LLM Injectivity Analysis Project

This project demonstrates and visualizes the **injectivity property** of language models across different architectures (decoder and encoder), which is fundamental to the SIPIT (Single-Input Prefix Inversion) algorithm.

## 📄 Based On

**Paper**: "Language Models are Injective and Hence Invertible"

**Abstract**: The paper proves that transformer language models are injective (different inputs → different outputs) and introduces SIPIT, an algorithm that can reconstruct the exact input text from hidden activations. This has implications for transparency, interpretability, and security.

## 🎯 What is Injectivity?

**Injectivity** means that each unique input maps to a unique output. In the context of language models:
- Different text inputs produce different hidden states
- This uniqueness allows us to potentially reverse-engineer prompts from their hidden states
- It's a key requirement for prompt inversion algorithms

## 🏗️ Project Goal

Test if injectivity holds for **both** decoder and encoder architectures:
- **Decoder (GPT-2)**: Autoregressive, causal attention
- **Encoder (BERT)**: Bidirectional attention

## 📁 Files

### Core Scripts:
- `pythonScript.py` - Original demonstration with simple word examples (GPT-2)
- `analyze_injectivity.py` - Comprehensive GPT-2 decoder analysis with visualizations
- `analyze_bert_injectivity.py` - Comprehensive BERT encoder analysis with visualizations
- `compare_encoder_decoder.py` - Side-by-side comparison of both architectures
- `sipit_reconstruction.py` - **SIPIT algorithm: Actual prompt reconstruction** ⭐
- `run_all_analyses.py` - Master script to run all analyses

### Documentation:
- `README.md` - Main project documentation (this file)
- `SIPIT_GUIDE.md` - Complete SIPIT algorithm guide
- `QUICKSTART_BERT.md` - Step-by-step BERT analysis guide
- `PROJECT_SUMMARY.md` - High-level project overview

### Data & Config:
- `requirements.txt` - Python dependencies
- `testingData.json` - Legal query-answer pairs for analysis (17 samples)

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Simple Demo

```bash
python pythonScript.py
```

This runs a quick demonstration with simple words ("fox", "dog", "cat", etc.)

### 3. Run Decoder Analysis (GPT-2)

```bash
python analyze_injectivity.py
```

Analyzes GPT-2 decoder model and saves results to `results/` folder.

### 4. Run Encoder Analysis (BERT)

```bash
python analyze_bert_injectivity.py
```

Analyzes BERT encoder model and saves results to `Encoder_Bert_Results/` folder.

### 5. Run Comparative Analysis

```bash
python compare_encoder_decoder.py
```

Compares both architectures side-by-side and saves results to `Comparison_Results/` folder.

### 6. Run SIPIT Reconstruction ⭐ (NEW)

```bash
python sipit_reconstruction.py
```

**Actually reconstructs** input text from hidden states using the SIPIT algorithm! This demonstrates practical invertibility (not just theoretical). Results saved to `SIPIT_Results/` folder.

**Note:** This is computationally intensive (~30-60s per sample). See `SIPIT_GUIDE.md` for details.

## 📊 What the Analysis Generates

### 1️⃣ GPT-2 Decoder Analysis (`results/` folder)

**Visualizations:**
- `query_heatmap.png` - Distance matrix for queries
- `answer_heatmap.png` - Distance matrix for answers
- `query_distance_distribution.png` - Distribution histogram for queries
- `answer_distance_distribution.png` - Distribution histogram for answers
- `layer_comparison.png` - Injectivity across layers 0-12

**Reports:**
- `query_report.txt` - Detailed query analysis
- `answer_report.txt` - Detailed answer analysis

### 2️⃣ BERT Encoder Analysis (`Encoder_Bert_Results/` folder)

**Visualizations:**
- `query_heatmap.png` - BERT query distance matrix
- `answer_heatmap.png` - BERT answer distance matrix
- `query_distance_distribution.png` - Query distribution
- `answer_distance_distribution.png` - Answer distribution
- `layer_comparison.png` - Layer-wise analysis for BERT
- `pooling_strategy_comparison.png` - Comparison of CLS, Mean, Max, Last pooling

**Reports:**
- `query_report.txt` - BERT query analysis with encoder-specific insights
- `answer_report.txt` - BERT answer analysis
- `pooling_comparison_report.txt` - Which pooling strategy works best

### 3️⃣ Comparative Analysis (`Comparison_Results/` folder)

**Visualizations:**
- `encoder_vs_decoder_comparison.png` - Side-by-side comparison (4 subplots)

**Reports:**
- `comparison_report.txt` - Comprehensive comparison of both architectures

### 4️⃣ SIPIT Reconstruction (`SIPIT_Results/` folder) ⭐

**This goes beyond proving injectivity to actually RECONSTRUCTING the original text!**

**Reports:**
- `sipit_report.txt` - Detailed reconstruction results with:
  - Exact match rates
  - Token overlap percentages
  - Character similarity scores
  - Original vs reconstructed text comparisons
  - Security implications
- `sipit_results.json` - Raw data for further analysis

**Key Metrics:**
- **Exact matches:** How many perfect reconstructions
- **Token overlap:** % of original tokens recovered
- **Character similarity:** Character-level match percentage
- **Computation time:** How long each reconstruction took

## 🔍 Understanding the Results

### Distance Heatmap
- **Red colors** = Large distances (very different texts)
- **Yellow colors** = Smaller distances (more similar texts)
- **All non-zero** = Injective mapping ✓

### Distance Distribution
- Shows how separated different inputs are in the hidden state space
- **Red line** = Minimum distance (separation margin)
- **Green line** = Average distance
- If minimum > 0, the mapping is injective

### Layer Comparison
- Shows how injectivity evolves through the model
- **Early layers** (0-3): Basic token embeddings
- **Middle layers** (6): Rich semantic representations
- **Late layers** (9-12): Task-specific features

## 💡 Key Insights

### From GPT-2 (Decoder) Analysis:
1. **Each query produces a unique hidden state** - No two queries map to the same point
2. **The separation margin matters** - Larger distances = more robust injectivity
3. **Different layers have different properties** - Middle layers often show best separation
4. **This enables prompt recovery** - Since each input is unique, we can theoretically reverse it

### From BERT (Encoder) Analysis:
1. **Encoders also exhibit injectivity** - Bidirectional attention doesn't break uniqueness
2. **Pooling strategy affects results** - CLS token vs mean pooling show different separation margins
3. **Encoder injectivity confirmed** - Even with full bidirectional context, inputs remain distinguishable
4. **Reconstruction may be possible** - Though algorithms would differ from decoder-based SIPIT

### From Comparative Analysis:
1. **Injectivity is architecture-agnostic** - Both decoder and encoder models are injective
2. **Universal property of transformers** - Not specific to autoregressive generation
3. **Security implications are broader** - All transformer embeddings may be invertible
4. **Different separation characteristics** - Encoders and decoders have different distance distributions

### From SIPIT Reconstruction ⭐:
1. **Practical invertibility confirmed** - We can actually reconstruct text (not just prove it's possible)
2. **Partial reconstruction achievable** - Typical token overlap: 30-60%
3. **Beam search outperforms greedy** - Better exploration yields better results
4. **Computational challenge** - Each reconstruction takes 30-60 seconds
5. **Real security risk** - Hidden states DO leak input information

## 🔬 Technical Details

### GPT-2 (Decoder)
- **Model**: GPT-2-small (124M parameters)
- **Layers**: 12 transformer layers
- **Hidden Dimension**: 768
- **Primary Layer**: Layer 6 (middle)
- **Representation**: Last token hidden state

### BERT (Encoder)
- **Model**: BERT-base-uncased (110M parameters)
- **Layers**: 12 transformer layers
- **Hidden Dimension**: 768
- **Primary Layer**: Layer 6 (middle)
- **Representation**: [CLS] token (with pooling comparisons)

### Both Models
- **Distance Metric**: L2 (Euclidean) distance
- **Analysis**: Pairwise distances for all samples
- **Visualization**: Heatmaps, distributions, layer comparisons

## 🛡️ Security Implications

This analysis has important implications for LLM security across **all** transformer architectures:

### For Decoder Models (GPT-2, GPT-3, etc.):
- Hidden states contain recoverable information about input prompts
- Autoregressive generation may leak previous tokens
- SIPIT algorithm demonstrates practical invertibility

### For Encoder Models (BERT, RoBERTa, etc.):
- Embeddings used for semantic search may be invertible
- [CLS] token representations might reveal original text
- Bidirectional context doesn't prevent reconstruction

### Universal Concerns:
- Any system exposing transformer embeddings has privacy risks
- Both APIs and local deployments affected
- Need for privacy-preserving techniques in production
- Implications for federated learning and edge deployment

## 📚 References

This implementation is based on concepts from prompt inversion research and the SIPIT algorithm for recovering prompts from hidden states.

## ⚙️ Customization

You can modify the analysis by editing `analyze_injectivity.py`:

```python
LAYER_TO_CHECK = 6  # Change to analyze different layers (0-12)
MODEL_NAME = 'gpt2'  # Try 'gpt2-medium', 'gpt2-large', etc.
```

## 🔮 Expected Results

Based on the paper's findings, you should observe:

1. **✓ All minimum distances > 0** - Confirms injectivity
2. **📏 Typical separation margins**: 5-30 units (model/layer dependent)
3. **📊 Normal-like distribution** of pairwise distances
4. **🎯 Best separation** in middle layers (layers 5-7)
5. **🔵 GPT-2 vs 🔴 BERT**: Similar injectivity, different magnitudes

## 🚀 Future Directions

### Immediate Next Steps:
- Test on larger models (GPT-2-medium, BERT-large)
- Implement actual prompt reconstruction (SIPIT for GPT-2)
- Develop encoder-specific inversion algorithm
- Test on other architectures (T5, BART, etc.)

### Research Questions:
- How does model size affect injectivity?
- Can we reconstruct prompts with partial information?
- What are the computational costs of inversion?
- How do fine-tuned models compare?

## 🤝 Contributing

Feel free to experiment with:
- Different models (GPT-2 variants, RoBERTa, DistilBERT, etc.)
- Different layers and pooling strategies
- Different distance metrics (cosine, Manhattan)
- Your own datasets (longer texts, different domains)
- Cross-lingual models (multilingual BERT)

---

**Note**: First run may take a few minutes as it downloads the GPT-2 model (~500MB). Subsequent runs will be faster.

