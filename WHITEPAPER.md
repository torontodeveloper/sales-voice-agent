# ABC Energy Sales Voice Agent — Technical Whitepaper                                                              
                                                                                                                     
  ## Overview                                                                                                        
  This whitepaper covers the technical architecture, design decisions, and production        scaling plan for the ABC Energy Sales Voice Agent. It complements the README with additional details.

  This project uses my experience building a two-stage web agent on the   Mind2Web dataset at CMU/Fleetworthy, applying similar data normalization, fine-tuning, and
  evaluation patterns to a production voice sales context and ChatOPG at Ontario Power Generation(OPG). ChatOPG is chatbot user interfacet talking to GPT3.4, LLam,Mistral, Gemini LLMs which we cal them as bot as per user choice of bot                                         
  ---                                                                                                                
  ## 1. Data Engineering

  ### Dataset Strategy
  Combined two sources:
  - **SGD (Schema-Guided Dialogue)** — 14,919 task-oriented dialogues for natural               conversation structure via Mediform/sgd-sharegpt (function_cot_nlg config)                                       
  - **Synthetic Energy Dialogues** — 480 GPT-4o-mini generated records covering          objection handling, positive sales scenarios, and closing techniques. I nitially, I had 48 samples, then later 160 samples and finally to 480 sample synthetic data. I like to expand this to 1000s of record with proportin of 20% synthetic and 80% of SGD data with SGD data at 18k recores that brings roughly 3600 or so many synthetic records                                            

  ### Why function_cot_nlg                                                                                           
  Three SGD configs were evaluated:                                                                                  
  - `function_only` — raw function calls, no natural language                                                        
  - `function_cot` — chain-of-thought reasoning                                                                      
  - `function_cot_nlg` — CoT + natural language response ← chosen                                                    
  function_cot_nlg produces responses closest to natural voice agent output —                                        
  the assistant reasons through function calls then responds conversationally.                                       
                                                                                                                     
  ### Data Quality Challenges
  - GPT-4o-mini occasionally generates malformed JSON (mismatched quotes,                   string instead of dict for conversation turns)                                                                   
  - Fixed with validation filter at load time — reject any record where                       conversation turns are not all dicts                                                                             
  - SGD records use `from`/`value` keys vs Llama's expected `role`/`content` —                mapped at training  time                                                                                          
  ### Data Versioning                                                                                                
  Production would use **DVC (Data Version Control)** to version datasets without                                    
  storing large files in git. Current implementation uses .gitignore for data/ folder.                               
  ---                                                                                                                
  ## 2. Fine-Tuning & Alignment                                                                                      
  
  ### SFT (Supervised Fine-Tuning) — Completed                                                                       
  - **Model:** Llama 3.1 8B (via Unsloth)
  - **Method:** QLoRA — NF4 quantized base model + LoRA adapters in bfloat16                                         
  - **LoRA config:** r=16, alpha=16, target modules: q_proj, k_proj, v_proj, o_proj                                  
  - **Training:** 60 steps(this is minmial), Training loss: 1.47 → 0.40 (final step), avg 0.64 over 60 steps                          
  - **GPU:** NVIDIA L4 (22GB), load_in_4bit=True reduced memory from ~16GB to ~6GB                                   
  - **Parameters trained:** 41M of 8B total (0.52%)                                                                  
                                                                                                                     
  ### Why QLoRA over Full Fine-Tuning                                                                                
  Full fine-tuning Llama 3.1 8B requires 8× H100s (~$50K/month). QLoRA achieves                comparable results on a single L4 GPU ($2/hour on Colab) by:                                                       
  1. Loading frozen base weights in NF4 4-bit format                                                                 
  2. Training only low-rank adapter matrices (r=16 adds ~41M trainable params)                                       
  3. Using paged AdamW optimizer to handle memory spikes                                                             
                                                                                                                     
  ### DPO (Direct Preference Optimization) — Implemented, Blocked on Colab                                           
  - **Data:** 50 chosen/rejected pairs generated by GPT-4o-mini covering 10                                          
    objection scenarios × 5 variations                                                                               
  - **Config:** beta=0.1, lr=5e-6, 1 epoch, batch_size=2
  - **Blocked by:** trl/unsloth/transformers version incompatibility on Colab                                        
  - **Production fix:** Pinned environment with unsloth==2025.11.7, trl==0.11.4,                                     
    transformers==4.44.2 in Docker container                                                                         
                                                                                                                     
  ### Why DPO over RLHF                                                                                              
  - RLHF requires training a separate reward model then PPO optimization — 
    unstable, complex, expensive                                                                                     
  - DPO directly optimizes on preference pairs — simpler, more stable, 
    same alignment quality                                                                                           
  - Beta=0.1 controls deviation from reference model — lower = safer, 
    higher = more aggressive alignment                                                                               

  - We can also try RLHF for specific call scenarios or any tool calling                                                                                                                   
  ---                                                                                                                
  ## 3. Quantization & Serving

  ### vLLM Serving — Completed
  - Deployed Llama 3.1 8B Instruct via vLLM on Colab L4
  - Public endpoint via ngrok tunnel                                                                                 
  - FastAPI wrapper for sales agent API                                                                              
  - Config: gpu_memory_utilization=0.85, max_model_len=2048, enforce_eager=True                                      
                                                                                                                     
  ### Why vLLM    
  - **PagedAttention** — manages KV cache like OS virtual memory, eliminates                                         
    memory fragmentation, enables higher throughput                                                                  
  - **Continuous batching** — processes requests as they arrive vs. waiting
    for full batch, reduces latency                                                                                  
  - **OpenAI-compatible API** — drop-in replacement, no client code changes
                                                                                                                     
  ### GGUF Quantization — Documented
  ```bash                                                                                                            
  # Production command (Unsloth)                                                                                     
  model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")
  - Reduces model from ~16GB (FP16) to ~4.5GB (4-bit)                                                                
  - q4_k_m — best quality/size balance                                                                               
  - Deploy via llama.cpp for CPU inference or edge deployment                                                        
  - Trade-off: ~1-2% quality loss vs 4× memory reduction                                                             
                                                                                                                     
  AWQ vs GGUF vs NVFP4                                                                                               
                                                                                                                     
  ┌─────────────┬───────┬─────────┬─────────────┐                                                                    
  │   Format    │ Size  │ Quality │  Best For   │                                                                    
  ├─────────────┼───────┼─────────┼─────────────┤
  │ GGUF q4_k_m │ 4.5GB │ Good    │ CPU/edge    │
  ├─────────────┼───────┼─────────┼─────────────┤
  │ AWQ         │ 4.5GB │ Better  │ GPU serving │                                                                    
  ├─────────────┼───────┼─────────┼─────────────┤
  │ NVFP4       │ 4GB   │ Best    │ H100 only   │                                                                    
  └─────────────┴───────┴─────────┴─────────────┘                                                                    
  
  ---                                                                                                                
  4. Evaluation   
               
  LLM-as-a-Judge
                                                                                                                     
  GPT-4o-mini scores fine-tuned Llama responses on:                                                                  
  - Relevance (1-10) — addresses customer concern                                                                    
  - Persuasiveness (1-10) — moves toward sale                                                                        
  - Empathy (1-10) — acknowledges customer feelings
  - Accuracy (1-10) — correct energy product info                                                                    
  - Overall (1-10)                                                                                                   
   
  Eval Lessons                                                                                                       
                  
  Initial scores were 2-4/10 — diagnosed as evaluating SGD non-energy records                                        
  with energy sales criteria. Fixed by filtering eval set to synthetic energy
  records only. Rescored 6-8/10 range.                                                                               
                                                                                                                     
  ---                                                                                                                
  5. Production Scale (Bonus)                                                                                        
                             
  1000 Concurrent Voice Sessions
                                                                                                                     
  Voice calls require <500ms latency per turn. At 1000 concurrent sessions:                                          
                                                                                                                     
  Architecture:                                                                                                      
  Load Balancer (nginx)
      ├── vLLM Instance 1 (A100 80GB) — 300 concurrent
      ├── vLLM Instance 2 (A100 80GB) — 300 concurrent
      └── vLLM Instance 3 (A100 80GB) — 400 concurrent                                                               
                                                                                                                     
  Key vLLM features for scale:                                                                                       
  - PagedAttention — each session gets isolated KV cache pages,                                                      
  no memory fragmentation across concurrent requests                                                                 
  - Continuous batching — new requests join in-flight batches,                                                       
  no waiting for batch completion                                                                                    
  - Tensor parallelism — --tensor-parallel-size 2 splits model                                                       
  across 2 GPUs for larger batch sizes                        
                                                                                                                     
  Estimated capacity per A100 80GB:                                                                                  
  - Llama 3.1 8B in FP16 = ~16GB model weights                                                                       
  - Remaining 64GB for KV cache                                                                                      
  - At max_seq_len=2048: ~300 concurrent sessions per GPU
                                                                                                                     
  Speculative Decoding                                                                                               
                                                                                                                     
  Reduces latency by 2-3× using a small draft model:                                                                 
                                                                                                                     
  # vLLM speculative decoding                                                                                        
  python -m vllm.entrypoints.openai.api_server \
      --model "meta-llama/Llama-3.1-8B-Instruct" \                                                                   
      --speculative-model "meta-llama/Llama-3.2-1B-Instruct" \
      --num-speculative-tokens 5                                                                                     
                  
  Draft model (1B) proposes 5 tokens, target model (8B) verifies in parallel —                                       
  same quality, 2-3× faster.
                                                                                                                     
  Locust Benchmarking
                                                                                                                     
  # scripts/bench/locust_bench.py                                                                                    
  from locust import HttpUser, task, between
  import json                                                                                                        
                  
  class SalesAgentUser(HttpUser):                                                                                    
      wait_time = between(1, 3)
                                                                                                                     
      @task       
      def sales_call(self):
          self.client.post(
              "/v1/chat/completions",
              json={                                                                                                 
                  "model": "meta-llama/Llama-3.1-8B-Instruct",
                  "messages": [                                                                                      
                      {"role": "system", "content": "You are an ABC Energy sales agent."},
                      {"role": "user", "content": "I want to switch my electricity provider"}                        
                  ],                                                                                                 
                  "max_tokens": 200                                                                                  
              },                                                                                                     
              headers={"Content-Type": "application/json"}
          )                                                                                                          
   
  # Run: locust -f scripts/bench/locust_bench.py --host http://localhost:8000                                        
  # Target: 1000 users, ramp up 10/second
                                                                                                                     
  Key metrics to track:                                                                                              
  - p50/p95/p99 latency (target: p95 < 500ms for voice)                                                              
  - Requests/second throughput                                                                                       
  - Error rate under load
                                                                                                                     
  ---             
  6. Observability

  - Sentry =Sanity Testing- used Sentry at Fleetworthy during production deployment to verify deployment succeede and App Epic is Up and running again without any issues  
  - Grafana/Datadog also can be used for dashboard metrics such as GPU/CPU/memory footprint which I have used at Fleetworthy
  - Splunk,Kibana can also be used for Observability, Logging, Proactive Performance Monitoring(PPM)
                  
  MLflow (Training)
                                                                                                                     
  Tracks SFT/DPO experiments — loss curves, hyperparameters, model versions.                                         
  mlflow.set_experiment("sales-voice-agent-sft")                                                                     
  mlflow.autolog()  # auto-logs HuggingFace trainer metrics                                                          
                                                           
  LangSmith (Inference)                                                                                              
                                                                                                                     
  LLM-specific tracing for every vLLM call — prompt, response, latency, tokens.                                      
  from langsmith import traceable                                                                                    
                                                                                                                     
  @traceable      
  def call_agent(messages):
      return client.chat.completions.create(...)
  Captures conversation-level traces — which objections cause failures,                                              
  which turns have high latency (critical for voice <500ms SLA).       
                                                                                                                     
  Why Both                                                                                                           
                                                                                                                     
  - MLflow = training observability (offline)                                                                        
  - LangSmith = inference observability (online/production)                                                          
  - Complementary layers, not alternatives      
                                                    
   
  ---                                                                                                                
  7. Production Architecture
                            
  Customer Call
      → Speech-to-Text (Deepgram/Whisper)                                                                            
      → FastAPI (session management, history)
      → vLLM cluster (Llama 3.1 8B, PagedAttention)                                                                  
      → Text-to-Speech (ElevenLabs/Azure)                                                                            
      → Customer hears response                                                                                      
                                                                                                                     
  Observability:  
      → LangSmith (conversation traces)                                                                              
      → MLflow (model versioning)
      → Grafana/Datadog (infrastructure)                                                                             
   
  ---                                                                                                                
  References      
            
  - QLoRA paper: https://arxiv.org/abs/2305.14314
  - DPO paper: https://arxiv.org/abs/2305.18290                                                                      
  - vLLM PagedAttention: https://arxiv.org/abs/2309.06180                                                            
  - SGD Dataset: https://arxiv.org/pdf/1909.05855                                                                    
  - Unsloth: https://github.com/unslothai/unsloth                                                                    
  - TRL: https://github.com/huggingface/trl