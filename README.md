# TCD-SLP-Dissertation
# From Detection to Revision: Interpretable Approaches to Translationese

This repository contains code, trained models, and data resources developed as part of the M.Phil. dissertation project **"From Detection to Revision: Interpretable Approaches to Translationese in Chinese–English Literary Texts"** (Trinity College Dublin, 2025).

## Overview
The project investigates **translationese**—systematic linguistic patterns that differentiate translated texts from originals—and develops an **end-to-end pipeline** that:
1. Detects translationese in Chinese–English literary texts  
2. Explains model decisions using interpretable AI (SHAP, LIME)  
3. Generates targeted post-editing prompts for stylistic improvement  

## Contents
- **corpora/**  
  - *Translated English Corpus (TEC)*: Six Chinese novels with official English translations  
  - *Native English Corpus (NEC)*: 60 genre-matched English novels (Project Gutenberg)  
- **features/**  
  Scripts for extracting 14 handcrafted linguistic features (sentence length, clause depth, connectives, etc.) using spaCy/Stanza  
- **models/**  
  - Feature-based classifiers (Logistic Regression, Random Forest, MLP)  
  - Fine-tuned BERT model  
  - Training/evaluation scripts  
- **trained_models/**  
  Pre-trained and fine-tuned classifiers 
- **post_editing/**  
  Explanation-driven LLM rewriting pipeline (prompt templates, issue–solution mapping)  
- **results/**  
  Evaluation logs, COMETKiwi/BLEU scores, and interpretability outputs  

## Key Contributions
- A **genre-matched bilingual corpus** (TEC vs. NEC) for translationese analysis  
- Integration of **linguistic features + neural embeddings** in classification  
- **Transparent explanations** of translationese via SHAP/LIME  
- A proof-of-concept **post-editing assistant** linking detection → explanation → revision  

## Citation
If you use this repository, please cite:  
> Liu, Yuxuan. (2025). *From Detection to Revision: Interpretable Approaches to Translationese in Chinese–English Literary Texts*. M.Phil. Dissertation, Trinity College Dublin.

## License
This project is released under the MIT License.  


