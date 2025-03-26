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
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the test.tsv file
test_file_path = "/dgx1data/aii/zong/m322228/rag_patient_matching/dataset_trialgpt/sigir/qrels/test.tsv"
test_data = pd.read_csv(test_file_path, sep="\t")

# Load the retrieved_trials.json file
retrieved_trials_path = "/dgx1data/aii/zong/m322228/rag_patient_matching/dataset_trialgpt/sigir/retrieved_trials.json"
with open(retrieved_trials_path, "r") as file:
    retrieved_trials = json.load(file)

# Iterate over each patient and generate prompts based on whether they meet the criteria
# Create a dictionary to store trial details
trial_dict = {}
for trial in retrieved_trials:
    patient_id = trial["patient_id"]
    patient_context = trial["patient"]

    for trial_info in trial.values():
        if isinstance(trial_info, list):
            for trial_entry in trial_info:
                nct_id = trial_entry["NCTID"]
                trial_dict[nct_id] = {
                    "patient_id": patient_id,
                    "patient_context": patient_context,
                    "inclusion_criteria": trial_entry["inclusion_criteria"],
                    "exclusion_criteria": trial_entry["exclusion_criteria"]
                }

# Generate training dataset
training_samples = []
# Merge with test_data to get labels
for i, row in test_data.iterrows():
    # if i % 300 != 0:
    #     continue
    query_id = row["query-id"]
    corpus_id = row["corpus-id"]
    label = row["score"]

    if corpus_id in trial_dict:
        patient_id = trial_dict[corpus_id]["patient_id"]
        patient_context = trial_dict[corpus_id]["patient_context"]
        NCTID = corpus_id
        inclusion_criteria = trial_dict[corpus_id]["inclusion_criteria"]
        exclusion_criteria = trial_dict[corpus_id]["exclusion_criteria"]
        label = label
        # Sigir
        SYS_PROMPT = """### System:
                        You are an AI assistant evaluating clinical trial eligibility based on provided inclusion and exclusion criteria. Your task is to determine the patient's eligibility for the clinical trial and respond strictly with labels 0 or 1.

                        ---

                        ### Instructions:
                        1. **Check Inclusion Criteria**
                        The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.
                        2. **Check Exclusion Criteria**
                        The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.
                        3. **Final Eligibility Decision**
                        Determine the patient's eligibility for the criteria and output the label as the qualifying degree 0 or 1.
                        Note that  0 - would not refer this patient for this clinical trial; 1 - would consider referring this patient to this clinical trial upon further investigation or highly likely to refer this patient for this clinical trial.

                        ---

                        ### Response Format:
                        - Respond with a single number: 0 or 1.
                        - Do not include any additional text, explanations, or reasoning in your response.
                        \n\n
                        """
        # Trec 2021, Trec 2022
        # SYS_PROMPT = """### System:
        #                 You are an AI assistant evaluating clinical trial eligibility based on provided inclusion and exclusion criteria. Your task is to determine the patient's eligibility for the clinical trial and respond strictly with labels 0 or 1.
        #
        #                 ---
        #
        #                 ### Instructions:
        #                 1. **Check Inclusion Criteria**
        #                 The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.
        #                 2. **Check Exclusion Criteria**
        #                 The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.
        #                 3. **Final Eligibility Decision**
        #                 Determine the patient's eligibility for the criteria and output the label as the qualifying degree 0 or 1.
        #                 Note that  0 - the patient is not relevant for the trial in any way; 1 - the patient has the condition that the trial is targeting, but the exclusion criteria make the patient ineligible or the patient is eligible to enroll in the trial.
        #
        #                 ---
        #
        #                 ### Response Format:
        #                 - Respond with a single number: 0 or 1.
        #                 - Do not include any additional text, explanations, or reasoning in your response.
        #                 \n\n
        #                 """
        sys_query = (f"### Input: \nEligibility Criteria:\n{inclusion_criteria}\n{exclusion_criteria}\n\n")

        query = (
            f"Patient Context:\n{patient_context}\n\n"
            f"Based on the patient's information and the eligibility criteria, decide if the patient is eligible for the clinical trial.\n\n"
        )

        # Combine the system prompt, query, and context
        full_prompt = SYS_PROMPT + "\n" + sys_query + query + "\n\n### Response:\n"

        # Add to training samples
        if label == 0:
            training_samples.append({"prompt": full_prompt, "response": str(0)})
        elif label == 1 or label == 2:
            training_samples.append({"prompt": full_prompt, "response": str(1)})

        # model_id = "tiiuae/Falcon3-7B-Instruct"
        # model_id = "google/gemma-3-12b-it"
        # model_id = "MaziyarPanahi/Calme-7B-Instruct-v0.2"
        # model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        patient_tokens = len(tokenizer.encode(patient_context))
        print("patient_tokens", patient_tokens)

# Convert to DataFrame
df_fine_tune = pd.DataFrame(training_samples)

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ndcg_score
import numpy as np
from datasets import Dataset

# Custom evaluation function using model.generate()
def evaluate_model(model, tokenizer, val_data):
    """
    Evaluate the model using model.generate() for multi-class classification.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer associated with the model.
        val_data: Validation dataset.

    Returns:
        Evaluation metrics including TP, FP, FN per class.
    """
    pred_responses = []
    label_responses = []

    for _, val in val_data.iterrows():
        inputs = tokenizer(val["prompt"], return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=7,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the final numeric label (0 or 1) from the last non-empty line
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
        predicted_label = lines[-1].lower() if lines else "unknown"
        # # Extract prediction from generated text
        # predicted_label = generated_text.split(":")[-1].strip().lower()
        print(predicted_label)

        # Convert prediction into one of the 3 labels (0, 1, or 2)
        if '0' in predicted_label:
            pred_response = 0
        elif '1' in predicted_label:
            pred_response = 1
        # elif '2' in predicted_label:
        #     pred_response = 2
        else:
            pred_response = -1  # Default case if the model generates something unexpected

        label_response = val["response"]
        pred_responses.append(pred_response)
        label_responses.append(label_response)

    label_responses = [int(label) for label in label_responses]
    # Remove invalid predictions (-1)
    valid_indices = [i for i in range(len(pred_responses)) if pred_responses[i] != -1]
    pred_responses = [pred_responses[i] for i in valid_indices]
    label_responses = [label_responses[i] for i in valid_indices]

    print("Predicted responses:", pred_responses)
    print("True labels:", label_responses)

    # Compute evaluation metrics
    accuracy = accuracy_score(label_responses, pred_responses)
    precision = precision_score(label_responses, pred_responses, zero_division=0)
    recall = recall_score(label_responses, pred_responses, zero_division=0)
    f1 = f1_score(label_responses, pred_responses, zero_division=0)

    # Compute Micro and Macro F1 scores
    micro_f1 = f1_score(label_responses, pred_responses, average="micro", zero_division=0)
    macro_f1 = f1_score(label_responses, pred_responses, average="macro", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(label_responses, pred_responses).ravel()

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "true_labels": label_responses,
        "predicted_labels": pred_responses,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_index, val_index) in enumerate(kf.split(df_fine_tune)):
    # Model and tokenizer initialization
    # model_id = "tiiuae/Falcon3-7B-Instruct"
    # model_id = "google/gemma-3-12b-it"
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

    print(f"\n========== Fold {fold + 1} ==========")

    train_data = df_fine_tune.iloc[train_index]
    val_data = df_fine_tune.iloc[val_index]

    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Tokenize the datasets using the map function
    train_dataset = train_dataset.map(preprocess_data, remove_columns=["prompt", "response"], batched=False)
    val_dataset = val_dataset.map(preprocess_data, remove_columns=["prompt", "response"], batched=False)

    # Define SFTArguments
    sft_args = SFTConfig(
        output_dir=f"./checkpoints_fold_{fold}_0",
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
        logging_dir=f"./logs_fold_{fold}_0",
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
    fold_metrics.append(eval_metrics)

    print(f"Fold {fold + 1} Metrics:")
    for metric, value in eval_metrics.items():
        if isinstance(value, (list, dict)):  # Handle lists and dicts separately
            print(f"  {metric}: {value}")  # Print as-is
        else:
            print(f"  {metric}: {value:.4f}")  # Format only numeric values

    # Save fine-tuned model
    output_dir = f"./fine_tuned_peft_model_{fold}_0"
    trainer.model.save_pretrained(output_dir)

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map='auto', torch_dtype=torch.bfloat16)

    # Merge LoRA adapter weights with the base model
    model = model.merge_and_unload()

    # Save the complete model
    save_dir = f"./fine_tuned_complete_model_{fold}_0"
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    print(f"Fine-tuning Fold {fold + 1} completed, and the model is saved!")

    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()

print(f"\n========== Cross-Validation Results ==========")

# Initialize lists to store all true labels and predictions
all_true_labels = []
all_predicted_labels = []
# Initialize totals for counts across folds
total_tp = total_fp = total_fn = total_tn = 0

# Display fold-wise metrics
for fold, metrics in enumerate(fold_metrics, 1):
    fold_true_labels = metrics["true_labels"]
    fold_predicted_labels = metrics["predicted_labels"]

    # Append to global lists for later computation
    all_true_labels.extend(fold_true_labels)
    all_predicted_labels.extend(fold_predicted_labels)

    total_tp += metrics.get("tp", 0)
    total_fp += metrics.get("fp", 0)
    total_fn += metrics.get("fn", 0)
    total_tn += metrics.get("tn", 0)

    print(f"Fold {fold} Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (list, dict)):  # Handle lists and dicts separately
            print(f"  {metric}: {value}")  # Print as-is
        else:
            print(f"  {metric}: {value:.4f}")  # Format only numeric values

# Compute aggregated metrics using sklearn (global values)
accuracy = accuracy_score(all_true_labels, all_predicted_labels)
precision = precision_score(all_true_labels, all_predicted_labels, zero_division=0)
recall = recall_score(all_true_labels, all_predicted_labels, zero_division=0)
f1 = f1_score(all_true_labels, all_predicted_labels, zero_division=0)

# Compute Micro and Macro F1 using sklearn
micro_f1 = f1_score(all_true_labels, all_predicted_labels, average="micro", zero_division=0)
macro_f1 = f1_score(all_true_labels, all_predicted_labels, average="macro", zero_division=0)

all_true_labels = np.array(all_true_labels)  # Convert list to NumPy array
all_predicted_labels = np.array(all_predicted_labels)

# Flipping the dataset (0s become 1s, 1s become 0s)
flipped_true_labels = 1 - all_true_labels

flipped_predicted_labels = 1 - all_predicted_labels

print(all_true_labels)

print(all_predicted_labels)

print(flipped_true_labels)

print(flipped_predicted_labels)

# Compute AUROC
auroc_orig = roc_auc_score(all_true_labels, all_predicted_labels)
auroc_flipped = roc_auc_score(flipped_true_labels, flipped_predicted_labels)

# Compute NDCG@10
ndcg_10 = ndcg_score([all_true_labels], [all_predicted_labels], k=10)

# Compute P@10 (Precision at 10)
p_at_10 = np.sum(all_true_labels[:10] == all_predicted_labels[:10]) / 10

print("\nAggregated Metrics Across Folds:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-score: {f1:.4f}")
print(f"  Micro F1: {micro_f1:.4f}")
print(f"  Macro F1: {macro_f1:.4f}")
print(f"  auroc_orig: {auroc_orig:.4f}")
print(f"  auroc_flipped: {auroc_flipped:.4f}")
print(f"  ndcg_10: {ndcg_10:.4f}")
print(f"  p_at_10: {p_at_10:.4f}")
print(f"  total tp: {total_tp:.4f}")
print(f"  total fp: {total_fp:.4f}")
print(f"  total fn: {total_fn:.4f}")
print(f"  total tn: {total_tn:.4f}")

