import torch
import sqlglot
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# ==========================================
# 1. Configuration & Setup
# ==========================================
MODEL_NAME = "ibm-granite/granite-4.0-350m-base"
OUTPUT_DIR = "granite-grpo-wikisql"
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 128

# Using a subset for faster iteration/debugging. 
# Remove split="train[:2000]" to use the full dataset.
dataset = load_dataset("json", data_files="https://huggingface.co/datasets/Salesforce/wikisql/resolve/main/data/train.jsonl", split="train[:2000]")
# ==========================================
# 2. Data Formatting
# ==========================================
def format_wikisql(example):
    """
    Formats the input to include the table schema.
    This is crucial so the model knows valid column names.
    """
    table_header = example["table"]["header"]
    question = example["question"]
    
    # We define a strict template. 
    # The <sql> tags help the reward function locate the answer easily.
    prompt = (
        f"Generate a SQL query to answer the question based on the table schema.\n"
        f"Schema: {table_header}\n"
        f"Question: {question}\n"
        f"Wrap your answer in <sql> tags. For example: <sql>SELECT * FROM table</sql>.\n"
        f"Answer:"
    )
    return {
        "prompt": prompt,
        # We store metadata for reward functions to use later
        "schema_cols": table_header 
    }

# Map the formatting function
dataset = dataset.map(format_wikisql)

# ==========================================
# 3. Reward Functions
# ==========================================

def format_reward_func(completions, **kwargs):
    """
    Reward 1: Format Compliance.
    Encourages the model to use the requested <sql> tags.
    """
    rewards = []
    for completion in completions:
        # Check if strict formatting tags are present
        if "<sql>" in completion and "</sql>" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0) 
    return rewards

def sql_syntax_reward_func(completions, **kwargs):
    """
    Reward 2: SQL Syntax Validity.
    Uses sqlglot to parse the query. 
    +2.0 for valid SQL, -1.0 for parsing errors.
    """
    rewards = []
    for completion in completions:
        try:
            # Extract content between tags
            if "<sql>" in completion and "</sql>" in completion:
                query = completion.split("<sql>")[1].split("</sql>")[0].strip()
            else:
                # Fallback: try to parse the raw completion if tags are missing
                query = completion.strip()
            
            # sqlglot.parse returns a list of expressions. If it doesn't fail, it's valid syntax.
            # We use read="sqlite" because WikiSQL is SQLite-based.
            sqlglot.parse(query, read="sqlite")
            rewards.append(2.0)
        except Exception:
            rewards.append(-1.0)
    return rewards

def schema_consistency_reward_func(prompts, completions, **kwargs):
    """
    Reward 3: Schema Consistency.
    Checks if the columns used in the query actually exist in the schema provided in the prompt.
    This specifically targets hallucination.
    """
    rewards = []
    
    # We need to extract the schema from the prompt to verify the completion
    for prompt, completion in zip(prompts, completions):
        try:
            # 1. Extract Schema from Prompt (Reverse engineering the prompt format)
            # Prompt format: "...Schema: ['col1', 'col2']\nQuestion..."
            schema_part = prompt.split("Schema: ")[1].split("\nQuestion")[0]
            # Convert string representation of list back to list (simplified)
            valid_cols = [c.strip().strip("'").strip('"') for c in schema_part.strip("[]").split(",")]
            valid_cols = set(c.lower() for c in valid_cols)

            # 2. Extract Query
            if "<sql>" in completion and "</sql>" in completion:
                query = completion.split("<sql>")[1].split("</sql>")[0].strip()
            else:
                query = completion.strip()

            # 3. Parse columns used in the query
            parsed = sqlglot.parse_one(query, read="sqlite")
            # Find all Column nodes in the abstract syntax tree
            used_cols = set(col.name.lower() for col in parsed.find_all(sqlglot.exp.Column))
            
            # 4. Calculate Reward
            # We check if *all* used columns are in the valid_cols set.
            # We allow '*' as it represents all columns.
            if "*" in used_cols: 
                used_cols.remove("*")
                
            if used_cols.issubset(valid_cols):
                rewards.append(1.5) # Bonus for respecting schema
            else:
                rewards.append(-0.5) # Penalty for hallucinating columns
                
        except Exception:
            # If parsing fails or regex fails, neutral/negative reward
            rewards.append(0.0)
            
    return rewards

# ==========================================
# 4. Model & Trainer Initialization
# ==========================================

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model
# Using bfloat16 for efficiency. 350M is small enough to not need 4-bit quantization usually.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LoRA Configuration
# Even for small models, LoRA is preferred for GRPO to stabilize training.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

# GRPO Config
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,               # Lower LR for stability
    per_device_train_batch_size=2,    # Small batch size for GRPO logic
    gradient_accumulation_steps=4,
    num_generations=4,                # Number of outputs to sample per prompt (the "Group")
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    report_to="none"                  # Set to "wandb" if you want tracking
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_func, 
        sql_syntax_reward_func, 
        schema_consistency_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    print("Starting GRPO Training for Text-to-SQL...")
    trainer.train()
    
    # Save the final adapter
    trainer.save_model(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")