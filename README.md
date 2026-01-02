<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)">
  </a>
</p>

<h1 align="center"><b>IE403.Q11 - Social Media Mining</b></h1>

# IE403 Course Project: Hate Speech Detection and Highlighting for Vietnamese with Rationale Extraction (HARE)

> This repository contains the full implementation of **HARE**, a framework designed to detect and explain hate speech in Vietnamese social media text. Developed for the course **IE403.Q11 – Social Media Mining** at the University of Information Technology (UIT – VNU-HCM).  
>  
> The project focuses on **Explainable AI (XAI)** by utilizing **Large Language Models (LLMs)** and **Chain-of-Thought (CoT)** prompting to not only classify hate speech but also extract rationales and implied statements. We leverage the **Qwen2.5-7B** model fine-tuned with **QLoRA** to achieve state-of-the-art performance. 



---

## Team Information
| No. | Student ID | Full Name | Role | Github | Email |
|----:|:----------:|-----------|------|--------|-------|
| 1 | 23521143 | Nguyen Cong Phat | Leader | [paht2005](https://github.com/paht2005) | 23521143@gm.uit.edu.vn |
| 2 | 23520032 | Truong Hoang Thanh An | Member | [Awnpz](https://github.com/Awnpz) | 23520032@gm.uit.edu.vn  |
| 3 | 23520023 | Nguyen Xuan An | Member | [annx-uit](https://github.com/annx-uit) |  23520023@gm.uit.edu.vn  | 
| 4 | 23520158 | Mai Thai Binh | Member | [maibinhkznk209](https://github.com/maibinhkznk209/) |  23520158@gm.uit.edu.vn  | 
| 5 | 21520255 | Nguyen Le Quynh Huong | Member | [tracycute](https://github.com/tracycute) |  21520255@gm.uit.edu.vn  | 


---


## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Demo](#demo)
- [Conclusion](#conclusion)
- [License](#license)

---

## Features
- **Hate Speech Classification:** Binary classification (Censored/Uncensored) for Vietnamese social media comments.
- **Rationale Extraction:** Automatically highlights specific spans of text that trigger the hate speech label.
- **Implied Statement Inference:** Decodes sarcasm and toxic metaphors into clear, literal statements.
- **Fine-tuned LLM:** Custom fine-tuning of **Qwen2.5-7B-Instruct** using **QLoRA** on a specialized rationale dataset.
- **Full-stack Application:**
  - **FastAPI Backend:** Supports streaming responses and real-time inference.
  - **React + Vite Frontend:** Interactive UI for visualizing highlighted toxic spans.

---

## Dataset
- **Core Dataset:** ViTHSD (Vietnamese Toxic Hate Speech Dataset).
- **Rationale Enhancement:** 2,333 samples were enriched with rationales and implied statements using GPT-4o.
- **Data splits:**
  - `dataset/raw/`: Original ViTHSD files in `.xlsx` format.
  - `data/processed/dataset_rationale.json`: The final processed dataset used for fine-tuning.

---

## Repository Structure
```
IE403.Q11_Hate-Speech-Detection-and-Highlighting-for-Vietnamese-Project/
├── dataset/
│   ├── raw/                        # Original ViTHSD .xlsx files
│   └── processed/                  # dataset_rationale.json for training
│
├── research/                       # Research and Model Development
│   ├── notebooks/                  # Training notebooks (Qwen, PhoBERT, Flan-T5)
│   ├── prompts/                    # Evolution of Prompt Engineering (v1, v2, Final)
│   └── src/                        # Modular scripts (config.py, models.py, eval.py)
│
├── demo/                           # Full Application (Stored on OneDrive)
│   ├── frontend/                   # React + Vite source code
│   ├── backend/                    # FastAPI server code
│   └── output/                     # Local logs and sample outputs
│
├── docs/                           # Documentation and Presentation
│   ├── IE403_Report.pdf            # Detailed academic report
│   └── IE403_Slide.pdf             # Presentation slides
│
├── requirements.txt                # Global Python dependencies
├── .gitignore                      # Git exclusion rules (Excludes 1.7GB demo folder)
└── README.md                       # Main project documentation

```

---
## Methodology

### 1. Fine-tuning Pipeline
We utilize **QLoRA (Quantized Low-Rank Adaptation)** to fine-tune **Qwen2.5-7B-Instruct** in a 4-bit quantized format, allowing high performance on consumer-grade GPUs.

### 2. Chain-of-Thought (CoT) Prompting
To improve the model's reasoning, we implemented a multi-stage prompt strategy:
- **Phase 1 (Rationale):** Identify why the text is toxic.
- **Phase 2 (Implied Statement):** Translate hidden toxic meanings.
- **Phase 3 (Labeling):** Final classification based on the extracted evidence.

### 3. Comparison Models (Baselines)
We compared our HARE framework against:
- **PhoBERT-base:** Traditional Encoder-only transformer.
- **Flan-T5-base:** Encoder-Decoder model for text-to-text tasks.

---

## Installation

### 1. Clone repository
```bash
git clone https://github.com/paht2005/IE403.Q11_Hate-Speech-Detection-and-Highlighting-for-Vietnamese-Project.git
cd IE403.Q11_Hate-Speech-Detection-and-Highlighting-for-Vietnamese-Project
```
### 2. (Optional) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows

```
### 3. Install dependencies
```bash
ppip install -r requirements.txt
```

### 4. Setup Demo (External Storage)
Due to the model weight size (~1.7GB), the `demo/` folder must be downloaded from OneDrive:
- **Link:** [Insert Your OneDrive Link Here]
- After downloading, extract the folder into the project root.

--- 

## Usage
### 1. Training & Research
Navigate to the research folder to reproduce experiments:
```bash
cd research/notebooks
jupyter notebook qwen_rationale.ipynb
```

### 2. Running the Demo
**Backend:**
```bash
pcd demo/backend
python main.py
```
**Frontend:**
```bash
cd demo/frontend
npm install
npm run dev

```

---
## Results

### Performance on ViTHSD Test Set

The table below compares HARE (Qwen2.5-7B) with baseline models.

| Model | Precision | Recall | F1-Score |
|------|-----|-----|------------|
| PhoBERT-base | 0.5097 | 0.5147 | 0.5122 |
| Flan-T5-base   | 0.4437 | 0.5348 | 0.4850 |
| HARE (Qwen2.5-7B Fine-tuned) | **0.6304** | **0.5772** | **0.6026** |


> **Key Finding:**  The inclusion of rationales and CoT reasoning improved the F1-score by nearly 10% compared to the best traditional transformer baseline.


---

### Comparison Summary


---

## Demo

The interactive web demo allows users to:
1. Input a Vietnamese social media comment.
2. View the real-time classification (Censored/Uncensored).
3. See **highlighted text spans** that the model identified as toxic.
4. Read the model's generated **explanation** for its decision.

---
## Conclusion
- Successfully built a Vietnamese hate speech detection system with high interpretability.
- Proved that LLMs fine-tuned with rationales significantly outperform traditional BERT-based models.
- Provided a modular framework for future research in Vietnamese XAI (Explainable AI).
 
---
## License
This project is for academic use in the course **IE403.Q11 - Social Media Mining** at the University of Information Technology (UIT – VNU-HCM).

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

