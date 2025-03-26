import os
import pandas as pd
import torch
from openpyxl.styles.builtins import total
from tensorflow.python.ops.gen_nn_ops import top_k
from torch import bfloat16
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.model_selection import KFold
import xml.etree.ElementTree as ET

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the final information to get the list of patients that meet the criteria
# Define directories
train_folder = "/dgx1data/aii/zong/m322228/rag_patient_matching/data_n2c2/train/"
test_folder = "/dgx1data/aii/zong/m322228/rag_patient_matching/data_n2c2/test_notags/"
gold_standard_folder = "/dgx1data/aii/zong/m322228/rag_patient_matching/data_n2c2/n2c2-t1_gold_standard_test_data/test/"

# Function to parse XML and extract patient text and labels
def extract_patient_data(xml_file, has_labels=True):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    text_element = root.find("TEXT")
    tags_element = root.find("TAGS")

    text = text_element.text.strip() if text_element is not None else ""
    labels = {}

    if has_labels and tags_element is not None:
        for tag in tags_element:
            labels[tag.tag] = tag.attrib.get("met", "not met")

    return text, labels

# Load training data
train_texts, train_labels = [], []
for file in os.listdir(train_folder):
    if file.endswith(".xml"):
        file_path = os.path.join(train_folder, file)
        patient_text, labels = extract_patient_data(file_path)
        train_texts.append(patient_text)
        train_labels.append(labels)

# Load test data without labels
test_texts, test_labels = [], []
for file in os.listdir(gold_standard_folder):
    if file.endswith(".xml"):
        file_path = os.path.join(gold_standard_folder, file)
        patient_text, labels = extract_patient_data(file_path)
        test_texts.append(patient_text)
        test_labels.append(labels)

def extract_tag(tag):
    if tag == "ABDOMINAL": tag_expand = "History of intra-abdominal surgery, small or large intestine resection, or small bowel obstruction"
    elif tag == "ADVANCED-CAD": tag_expand = "Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: • Taking 2 or more medications to treat CAD • History of myocardial infarction (MI)• Currently experiencing angina • Ischemia, past or present"
    elif tag == "ALCOHOL-ABUSE": tag_expand = "Current alcohol use over weekly recommended limits "
    elif tag == "ASP-FOR-MI": tag_expand = "Use of aspirin to prevent MI"
    elif tag == "CREATININE": tag_expand = "Serum creatinine > upper limit of normal "
    elif tag == "DIETSUPP-2MOS": tag_expand = "Taken a dietary supplement (excluding vitamin D) in the past 2 months"
    elif tag == "DRUG-ABUSE": tag_expand = "Drug abuse, current or past "
    elif tag == "ENGLISH": tag_expand = "Patient must speak English "
    elif tag == "HBA1C": tag_expand = "Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%"
    elif tag == "KETO-1YR": tag_expand = "Diagnosis of ketoacidosis in the past year "
    elif tag == "MAJOR-DIABETES": tag_expand = "Major diabetes-related complication. For the purposes of this annotation, we define “major complication” (as opposed to “minor complication”) as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes:• Amputation • Kidney damage • Skin conditions • Retinopathy • nephropathy • neuropathy"
    elif tag == "MAKES-DECISIONS": tag_expand = "Patient must make their own medical decisions "
    elif tag == "MI-6MOS": tag_expand = "MI in the past 6 months "

    return tag_expand

training_samples = []

# Initialize text splitter for patient data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# i = 0
# Iterate over each patient and generate prompts based on whether they meet the criteria
for row, labels in zip(train_texts, train_labels):
    # if i % 30 != 0:
    #     continue
    patient_text = row

    # Split text and create Document objects for the current patient
    splits = text_splitter.split_text(patient_text)
    patient_docs = [Document(page_content=split) for split in splits]

    # Generate embeddings and store in Chroma for the current patient
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                                       model_kwargs={"device": device})
    patient_vectordb = Chroma(
        persist_directory="chroma_patient_db_ac_5fold",
        embedding_function=embeddings,
        collection_name=f"patient_context"
    )
    patient_vectordb.add_documents(patient_docs)

    patient_retriever = patient_vectordb.as_retriever()

    for tag in labels.keys():
        decision = "Not Defined"

        # Check if the patient meets the criteria based on final_information
        if labels[tag] == "met":
            # Patient meets the criteria
            decision = "Yes"
        elif labels[tag] == "not met":
            # Patient does not meet the criteria
            decision = "No"

        if decision == "Yes" or decision == "No":
            SYS_PROMPT = """### System:
                            You are an AI assistant evaluating clinical trial eligibility based on provided criteria. Your task is to determine whether the patient is eligible for the clinical trial and respond strictly with "Yes" or "No."

                            ---

                            ### Instructions:
                            1. - If the given criteria is unsatisfied, respond:
                                 - "No"
                                 - End Task


                            2. - If the given criteria is satisfied, respond:
                                 - "Yes"
                                 - End Task

                            ---

                            ### Response Format:
                            - Respond with a single word: **"Yes"** or **"No"**.
                            - Do not include any additional text, explanations, or reasoning in your response.

                            ---

                            ### Important Notes:
                            - **Balance Fairness Between "Yes" and "No"**:
                              - If the patient meets the given criteria respond confidently with **"Yes"**.
                              - Do **not assume "No" as the default response** unless there is a **clear reason to reject eligibility**.

                            - **Do Not Over-Rely on Default "No" Responses**:
                              - **"Yes" is a valid answer** when criteria are met.
                              - If criteria are unclear, **retrieve additional data before defaulting to "No"**.

                            - **Maximize Recall Without Sacrificing Precision**:
                              - **Ensure all relevant patient context is retrieved.**
                              - Reduce **false negatives** by ensuring eligible patients are not mistakenly rejected.
                            \n\n
                            """

            tag_expand = extract_tag(tag)
            sys_query = (f"### Input: \nEligibility Criteria: {tag_expand}\n\n")
            # Retrieve relevant patient context
            # patient_retriever.search_kwargs["k"] = 1
            context_docs = patient_retriever.invoke(sys_query)
            # Validate the metadata of retrieved documents
            context = " ".join(doc.page_content for doc in context_docs)

            # print(context)

            query = (
                f"Patient Context:\n{context}\n\n"
                f"Based on the patient's information and the eligibility criteria, decide if the patient is eligible for the clinical trial.\n\n"
            )

            # Combine the system prompt, query, and context
            full_prompt = SYS_PROMPT + "\n" + sys_query + query + "\n\n### Response:\n"
            if decision == "Yes":
                response = f"{decision}"
            else:
                response = f"{decision}"

            # Add to training samples
            training_samples.append({"prompt": full_prompt, "response": response})

    # model_id = "tiiuae/Falcon3-7B-Instruct"
    # model_id = "google/gemma-2-9b-it"
    # model_id = "MaziyarPanahi/Calme-7B-Instruct-v0.2"
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    patient_tokens = len(tokenizer.encode(patient_text))
    print("patient_tokens", patient_tokens)

    # Ensure the collection is cleared
    patient_vectordb.delete_collection()
    # i += 1

testing_samples = []
# i = 0
# Iterate over each patient and generate prompts based on whether they meet the criteria
for row, labels in zip(test_texts, test_labels):
    # if i % 30 != 0:
    #     continue
    patient_text = row

    # Split text and create Document objects for the current patient
    splits = text_splitter.split_text(patient_text)
    patient_docs = [Document(page_content=split) for split in splits]

    # Generate embeddings and store in Chroma for the current patient
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                                       model_kwargs={"device": device})
    patient_vectordb = Chroma(
        persist_directory="chroma_patient_db_ac_5fold",
        embedding_function=embeddings,
        collection_name=f"patient_context"
    )
    patient_vectordb.add_documents(patient_docs)

    patient_retriever = patient_vectordb.as_retriever()

    for tag in labels.keys():
        decision = "Not Defined"

        # Check if the patient meets the criteria based on final_information
        if labels[tag] == "met":
            # Patient meets the criteria
            decision = "Yes"
        elif labels[tag] == "not met":
            # Patient does not meet the criteria
            decision = "No"

        if decision == "Yes" or decision == "No":
            SYS_PROMPT = """### System:
                            You are an AI assistant evaluating clinical trial eligibility based on provided criteria. Your task is to determine whether the patient is eligible for the clinical trial and respond strictly with "Yes" or "No."

                            ---

                            ### Instructions:
                            1. - If the given criteria is unsatisfied, respond:
                                 - "No"
                                 - End Task


                            2. - If the given criteria is satisfied, respond:
                                 - "Yes"
                                 - End Task

                            ---

                            ### Response Format:
                            - Respond with a single word: **"Yes"** or **"No"**.
                            - Do not include any additional text, explanations, or reasoning in your response.

                            ---

                            ### Important Notes:
                            - **Balance Fairness Between "Yes" and "No"**:
                              - If the patient meets the given criteria respond confidently with **"Yes"**.
                              - Do **not assume "No" as the default response** unless there is a **clear reason to reject eligibility**.

                            - **Do Not Over-Rely on Default "No" Responses**:
                              - **"Yes" is a valid answer** when criteria are met.
                              - If criteria are unclear, **retrieve additional data before defaulting to "No"**.

                            - **Maximize Recall Without Sacrificing Precision**:
                              - **Ensure all relevant patient context is retrieved.**
                              - Reduce **false negatives** by ensuring eligible patients are not mistakenly rejected.
                            \n\n
                            """

            tag_expand = extract_tag(tag)
            sys_query = (f"### Input: \nEligibility Criteria: {tag_expand}\n\n")
            # Retrieve relevant patient context
            # patient_retriever.search_kwargs["k"] = 1
            context_docs = patient_retriever.invoke(sys_query)
            # Validate the metadata of retrieved documents
            context = " ".join(doc.page_content for doc in context_docs)

            # print(context)

            query = (
                f"Patient Context:\n{context}\n\n"
                f"Based on the patient's information and the eligibility criteria, decide if the patient is eligible for the clinical trial.\n\n"
            )

            # Combine the system prompt, query, and context
            full_prompt = SYS_PROMPT + "\n" + sys_query + query + "\n\n### Response:\n"
            if decision == "Yes":
                response = f"{decision}"
            else:
                response = f"{decision}"

            # Add to training samples
            testing_samples.append({"prompt": full_prompt, "response": response})

    patient_tokens = len(tokenizer.encode(patient_text))
    print("patient_tokens", patient_tokens)

    # Ensure the collection is cleared
    patient_vectordb.delete_collection()
    # i += 1


# Convert to DataFrame
df_fine_tune_train = pd.DataFrame(training_samples)
df_fine_tune_test = pd.DataFrame(testing_samples)

def preprocess_data(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding=False)
    targets = tokenizer(examples["response"], truncation=True, padding=False)

    # Concatenate input and output
    input_ids = inputs["input_ids"] + targets["input_ids"]
    # Mask input tokens in labels with -100
    # labels = [-100] * len(inputs["input_ids"]) + targets["input_ids"]

    return {"input_ids": input_ids}

# Define Trainer
from transformers import Trainer

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from transformers import DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from datasets import Dataset

# Custom evaluation function using model.generate()
def evaluate_model(model, tokenizer, val_data):
    """
    Evaluate the model using model.generate().

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer associated with the model.
        val_data: Validation dataset.

    Returns:
        Evaluation metrics as a dictionary.
    """
    pred_responses = []
    label_responses = []

    for idx, val in val_data.iterrows():
        inputs = tokenizer(val["prompt"], return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=7, no_repeat_ngram_size=3, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract prediction from generated text
        # answer = generated_text.split(":")[-1].strip().lower()
        # Extract the final numeric label (0 or 1) from the last non-empty line
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
        answer = lines[-1].lower() if lines else "unknown"
        # generated_text = tokenizer.decode(outputs[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)
        # print(f"Generated Text: {generated_text}")

        # Extract decision from predictions and labels
        print("answer:", answer)
        if 'yes' in answer:
            pred_response = 'Yes'
        else:
            pred_response = 'No'

        label_response = val["response"]

        pred_responses.append(pred_response)
        label_responses.append(label_response)

    print("pred_responses", pred_responses)
    print("label_responses", label_responses)

    # Compute evaluation metrics
    binary_predictions = [1 if pred.lower() == "yes" else 0 for pred in pred_responses]
    binary_labels = [1 if label.lower() == "yes" else 0 for label in label_responses]

    accuracy = accuracy_score(binary_labels, binary_predictions)
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)

    # Compute Micro and Macro F1 scores
    micro_f1 = f1_score(binary_labels, binary_predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(binary_labels, binary_predictions, average='macro', zero_division=0)

    # Compute confusion matrix to extract TP, FP, FN, TN
    tn, fp, fn, tp = confusion_matrix(binary_labels, binary_predictions).ravel()

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "true_labels": binary_labels,
        "predicted_labels": binary_predictions,
    }

# Model and tokenizer initialization
# model_id = "tiiuae/Falcon3-7B-Instruct"
# model_id = "google/gemma-2-9b-it"
# model_id = "MaziyarPanahi/Calme-7B-Instruct-v0.2"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Set BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

print(f"Using device: {device}")

# Load model with BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    trust_remote_code=True,
    quantization_config=bnb_config,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Set padding token if missing
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token or '[PAD]'})
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Apply LoRA for fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For causal language modeling tasks
    r=4,  # Rank of LoRA updates
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # LoRA on attention layers
    lora_dropout=0.2,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print the number of trainable parameters

# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
response_template = "\n\n### Response:\n"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

train_data = df_fine_tune_train
val_data = df_fine_tune_test

# Create Hugging Face datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Tokenize the datasets using the map function
train_dataset = train_dataset.map(preprocess_data, remove_columns=["prompt", "response"], batched=False)
val_dataset = val_dataset.map(preprocess_data, remove_columns=["prompt", "response"], batched=False)

# Define SFTArguments
sft_args = SFTConfig(
    output_dir=f"./checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    warmup_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    bf16=torch.cuda.is_bf16_supported(),
    save_total_limit=2,  # Keep only last 2 checkpoints
    logging_dir=f"./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"  # Save based on eval loss
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,  # The pre-trained model
    train_dataset=train_dataset,  # Preprocessed training dataset
    eval_dataset=val_dataset,  # Preprocessed validation dataset
    args=sft_args,  # Training arguments using SFTArguments
    tokenizer=tokenizer,  # Tokenizer for tokenization and generation
    data_collator=data_collator,  # Collator for dynamic padding if needed
    dataset_text_field='input_ids',
    peft_config=lora_config,
    packing=False
)

# Train the model
trainer.train()

# Evaluate and retrieve metrics
# eval_metrics = trainer.evaluate()
eval_metrics = evaluate_model(trainer.model, tokenizer, val_data)

print(f"Metrics:")
for metric, value in eval_metrics.items():
    if metric not in ["true_labels", "predicted_labels"]:  # Exclude these keys
        print(f"  {metric}: {value:.4f}")

# Save fine-tuned model
output_dir = f"./fine_tuned_peft_model"
trainer.model.save_pretrained(output_dir)

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map='auto', torch_dtype=torch.bfloat16)

# Merge LoRA adapter weights with the base model
model = model.merge_and_unload()

# Save the complete model
save_dir = f"./fine_tuned_complete_model"
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)

print(f"Fine-tuning completed, and the model is saved!")

# Free memory
del model
del trainer
torch.cuda.empty_cache()