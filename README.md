# Sales Voice Agent

This pipeline architecture draws from my experience building a two-stage web agent on the Mind2Web dataset at CMU/Fleetworthy, where I applied similar data normalization, fine-tuning, and GPT-4o evaluation patterns.

ABC Energy is building a specialized LLM to power a next-generation Voice AI Sales Agent. The goal is to move beyond generic models and create a high-performance, domain-specific model capable of handling complex, real-time sales dialogues.

This builds a production-grade engineering loop: from raw data processing and fine-tuning to quantization and concurrency.

## Training Data

Training data combines real SGD task-oriented dialogues for natural conversation structure with GPT-4o synthesized energy sales scenarios for domain-specific objection handling and closing techniques.

## Design Decision

1. Added synthetic data generation of 8 energy-specific objection handling scenarios
2.Using Llama 3.1 8B base model, since there are more than 1000 records. As per Unsloth, a base model is a good option with a larger dataset, so we can fine-tune the LLM (Llama) on the given dataset.


## Enhancements

1. Right now, we only got dataset from Schema-Guided Dataset train only and augment with synthetic data, but we could use both train and test data and augment with synthetic before applying the following splits.

2. Train / validation / test data split:
   - 80 / 10 / 10
   - can be changed to 70 / 20 / 10
   - or 70 / 10 / 20

3. Right now, I am choosing open-source Llama 3.1 8B, but I would like to test Llama 4 as an enhanced version to check compatibility.


## References

### Schema-Guided Dialogue (SGD)
**Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset**

- https://arxiv.org/pdf/1909.05855
- https://huggingface.co/datasets/GEM/schema_guided_dialog
- https://huggingface.co/datasets/Mediform/sgd-sharegpt
- https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use