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

def preprocess_classification_data(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding=False)
    inputs["labels"] = examples["label"]
    return inputs

# Define Trainer
from transformers import Trainer

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ndcg_score
import numpy as np
from datasets import Dataset

# Custom evaluation function using model.generate()
def compute_metrics(eval_pred):
    """
    Evaluate the model using model.generate() for multi-class classification.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer associated with the model.
        val_data: Validation dataset.

    Returns:
        Evaluation metrics including TP, FP, FN per class.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    print("logits", logits)
    print("predictions", predictions)
    print("labels", labels)

    # Ensure all values are Python scalars (convert NumPy arrays to lists)
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    predictions = predictions.tolist() if hasattr(predictions, "tolist") else predictions

    # Compute evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Compute Micro and Macro F1 scores
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "true_labels": labels,
        "predicted_labels": predictions,
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
    model.config.problem_type = "single_label_classification"

    hidden_size = model.config.hidden_size
    num_classes = 2
    # model_with_classifier = LLaMAWithClassifier(base_model_with_lora, hidden_size, num_classes, True)
    classifier_device = next(model.parameters()).device
    model.classifier = torch.nn.Linear(hidden_size, num_classes).to(classifier_device)

    def new_forward(input_ids, attention_mask=None, labels=None, **kwargs):
        # Ensure inputs are moved to the first parameter's device
        device = next(model.parameters()).device
        # print(f"\n[DEBUG] Moving inputs to device: {device}")

        # Move inputs to the correct device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Forward pass
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Extract last hidden state
        hidden_states = outputs.hidden_states[-1]
        cls_token_output = hidden_states[:, -1, :]

        # Ensure classifier is on the same device as CLS output
        cls_token_output = cls_token_output.to(model.classifier.weight.device)
        logits = model.classifier(cls_token_output)

        if labels is not None:
            # Move loss to `cuda:0` before returning
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = loss.to("cuda:0")  # Move loss explicitly to `cuda:0`
            # print(f"\n[DEBUG] Moving loss to device: {loss.device}")
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    model.forward = new_forward

    data_collator = DataCollatorWithPadding(tokenizer)

    print(f"\n========== Fold {fold + 1} ==========")

    df_fine_tune["label"] = df_fine_tune["response"].map({"0": 0, "1": 1})
    train_data = df_fine_tune.iloc[train_index]
    val_data = df_fine_tune.iloc[val_index]

    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Tokenize the datasets using the map function
    train_dataset = train_dataset.map(preprocess_classification_data, remove_columns=["prompt", "response"], batched=False)
    val_dataset = val_dataset.map(preprocess_classification_data, remove_columns=["prompt", "response"], batched=False)

    # Define SFTArguments
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./checkpoints_classification_fold_{fold}_0",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        logging_dir=f"./logs_classification_fold_{fold}_0",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"  # Use evaluation loss for checkpoint saving
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate and retrieve metrics
    eval_metrics = trainer.evaluate()
    # eval_metrics = evaluate_model(trainer.model, tokenizer, val_data)
    fold_metrics.append(eval_metrics)

    print(f"Fold {fold + 1} Metrics:")
    for metric, value in eval_metrics.items():
        if isinstance(value, (list, dict)):  # Handle lists and dicts separately
            print(f"  {metric}: {value}")  # Print as-is
        else:
            print(f"  {metric}: {value:.4f}")  # Format only numeric values

    # Save fine-tuned model
    output_dir = f"./fine_tuned_peft_model_classification_{fold}_0"
    trainer.model.save_pretrained(output_dir)

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map='auto', torch_dtype=torch.bfloat16)

    # Merge LoRA adapter weights with the base model
    model = model.merge_and_unload()

    # Save the complete model
    save_dir = f"./fine_tuned_complete_model_classification_{fold}_0"
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
    fold_true_labels = metrics["eval_true_labels"]
    fold_predicted_labels = metrics["eval_predicted_labels"]

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


