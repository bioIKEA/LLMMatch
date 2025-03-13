# LLMMatch

This repository contains **four different code files** corresponding to **two methods** and **four datasets** for patient-trial matching using **LLM-Match**.  

## **Methods**  
We provide two different methods:  
1. **LLM-Match without classification head**  
2. **LLM-Match with classification head**  

## **Datasets**  
The code supports **four datasets**:  
- **n2c2 (2018 Cohort Selection Challenge)** → Uses **n2c2 code**  
- **SIGIR, TREC 2021, and TREC 2022** → Use **TrialGPT code**  

---  

## **Setup Instructions**  

Follow these steps to install dependencies, set up datasets, and run the models.  

### **1. Install Requirements**  
```bash  
pip install -r requirements.txt  
```  

### **2. Download and Define Dataset Paths**  
- **n2c2 Dataset**: Download from [n2c2 GitHub](https://github.com/som-shahlab/clinical_trial_patient_matching)  
- **SIGIR, TREC 2021, TREC 2022 Datasets**: Download from [TrialGPT GitHub](https://github.com/ncbi-nlp/TrialGPT)  

Define the path to your dataset in the corresponding code files.  

### **3. Run the Code**  
Use the appropriate script based on the dataset and method:  

- **For n2c2 dataset**:  
  ```bash  
  python rag_finetuning_ac_5fold_fixed_classification_n2c2.py
  ```
  
  or
  
  ```bash  
  python rag_finetuning_ac_5fold_fixed_n2c2.py  
  ```  

- **For SIGIR, TREC 2021, and TREC 2022 datasets**:  
  ```bash  
  python rag_finetuning_ac_5fold_fixed_classification_trialgpt.py
  ```
  
  or

  ```bash  
  rag_finetuning_ac_5fold_fixed_trialgpt.py
  ```

### **4. View Results**  
After execution, the evaluation results will be displayed at the end of the program run.
