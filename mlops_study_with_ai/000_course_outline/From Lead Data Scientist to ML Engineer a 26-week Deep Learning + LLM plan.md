# From Lead Data Scientist to ML Engineer: a 26-week Deep Learning + LLM plan

*15 h/week · about 390 h total · written 11 Nov 2026. Versions and links were checked on that date. This field moves monthly, so pin versions in a lockfile and skim release notes before each phase.*

## 0. How this plan works

### The 15-hour week (identical every week)

- **5 h Code-along:** replicate the video or repo line by line.
- **6 h Build:** the “mutate” step. Redo the core idea on your own data or model with the tutorial closed.
- **2 h Hardening:** tests, types, Dockerfile, a CI run, experiment tracking. This is what turns a tutorial into a portfolio piece.
- **1 h Cheat sheet:** read the matching one-pager (Appendix A) before the code-along, then write your own five-line version afterwards.
- **1 h Review:** run the weekly checklist (Section 7), commit, write one README paragraph.

### Replicate → Mutate → Rebuild

1. Type the tutorial code along with the instructor.
2. Change one thing that forces understanding: the dataset, the model size, the loss, or the framework.
3. A few days later, rebuild the core piece from a blank file. If you can’t, you copied it but haven’t learned it yet.

### One running domain

All fine-tuning, RAG and serving work uses a single domain so that every week compounds into the capstone: a **text-to-SQL analytics assistant**. It has public datasets (Spider, BIRD) and an objective metric (execution accuracy), which keeps evals honest. Swap the domain if you like, but keep an objective eval.

### Theory policy

Theory lives only in the 14 one-page cheat sheets in Appendix A. For everything else, learn by reading the code you are running. Items marked *(opt.)* in the schedule are the first to cut when you fall behind. Never cut Hardening.

### Compute and budget

- Weeks 1–8: Colab or Kaggle plus one rented 24 GB GPU (L4, A10 or 4090 class) covers everything.
- Weeks 13–22: new-account credits on AWS and GCP, plus hourly multi-GPU rentals (spot or preemptible where possible).
- My rough guess is $500–1,200 over six months, concentrated in Weeks 19–26. Treat it as a ballpark and check live prices.
- Day-1 rules: billing alarms on both clouds; everything created through Terraform or a script you can destroy; a “nothing left running” check at the end of every session.

### Three honest caveats

1. Big Tech ML engineer loops usually include data-structures-and-algorithms coding rounds as well as ML system design. This plan covers system design (the capstone, Weeks 25–26) but not algorithm drills. Budget 2–3 h/week on top, or accept the gap.
2. Lightning is on your priority list and is excellent for DL training templates, but most LLM fine-tuning in industry runs on Hugging Face Trainer/TRL with Accelerate or FSDP, or on custom loops. You will learn Lightning properly (Weeks 2 and 20) and also learn when to leave it.
3. Library churn is real. As of this date: LangChain is at v1 (agents are built with `create_agent`; legacy APIs moved to `langchain-classic`), TRL is at v1 with SFT, DPO, GRPO, RLOO and reward modeling as its stable trainers, MLflow 3 added GenAI tracing, evaluation and a prompt registry, and Lightning is on the 2.6 line. vLLM ships frequent releases, so pin it.

## 1. The tech stack

### Tier 1: your priority list (deep mastery)

| Tool | Master it by doing this | Weeks |
| --- | --- | --- |
| PyTorch 2.x | nn.Module and autograd, `torch.compile`, fused (flash) attention via SDPA, bf16 autocast, `torch.distributed`, DDP, FSDP2, the profiler | 1–4, 20–21 |
| PyTorch Lightning 2.6 | LightningModule, DataModule, Trainer, callbacks, DDP strategy, Fabric; plus when to drop to raw PyTorch | 2, 20 |
| Weights & Biases | runs, sweeps, Artifacts (datasets, models, lineage), Tables, Reports | 2 onward |
| MLflow 3 | tracking, model registry with aliases, GenAI tracing, `mlflow.genai.evaluate`, prompt registry; run the server behind auth (default installs have none) | 8, 12, 17 |
| Hugging Face | Transformers, Datasets, Tokenizers, Hub, PEFT, TRL v1 (SFT, DPO, GRPO), Accelerate, bitsandbytes | 5–8, 21 |
| vLLM | OpenAI-compatible server, benchmarking, quantization, LoRA serving, prefix caching, speculative decoding, tensor parallel | 18–19 |
| Ray | Core, Data (`ray.data.llm` batch inference), Serve, Train, KubeRay | 19, 22 |
| LangChain v1 (+ LangGraph) | `create_agent`, middleware, tools, structured output, LangGraph state machines | 11–12 |
| LlamaIndex | ingestion pipelines, retrievers, query engines, routers, metadata filters | 10–11 |
| Docker | multi-stage builds, CUDA images, BuildKit cache, compose, GPU runtime, image scanning | 13 |
| Kubernetes | Deployments, Services, Ingress, probes, HPA, Jobs, GPU scheduling, Helm | 14, 19, 22 |
| CI/CD | GitHub Actions as the primary tool: test, build, scan, eval gate, deploy; OIDC to both clouds | 1, 13, 17 |
| AWS | IAM, S3, ECR, VPC, EC2 GPU, ECS/EKS, SageMaker, Bedrock, CloudWatch | 15, 19, 22 |
| GCP | IAM, GCS, Artifact Registry, Cloud Run, Vertex AI, GKE (Ray Operator add-on), Cloud Monitoring | 16, 19, 22 |

### Tier 2: supporting tools (learn just enough, when needed)

uv, ruff, mypy or pyright, pytest, pre-commit; FastAPI, Pydantic, httpx; Terraform; Qdrant, pgvector, FAISS; Ragas, Langfuse, lm-evaluation-harness; DeepSpeed ZeRO; Unsloth and Axolotl; Prometheus and Grafana; Parquet/Arrow, datatrove, DVC; one orchestrator (Airflow or Prefect, pick one); Helm; k6 or Locust; Trivy; sqlglot (capstone).

### Tier 3: awareness only (no time budget)

SGLang, TensorRT-LLM, Triton and CUDA kernels (GPU MODE lectures), Kubeflow and KServe, Megatron-LM, JAX and TPUs.

## 2. Deep concepts, learned through code

Each row is something you write, not something you read. The C-numbers are referenced in the schedule.

| # | Concept | Implement this | “Done” means |
| --- | --- | --- | --- |
| C1 | Self-attention and multi-head attention | `MultiHeadAttention(nn.Module)` from scratch with a causal mask | Output matches `F.scaled_dot_product_attention` (`torch.allclose`); shape tests pass |
| C2 | Transformer block and GPT | Pre-LayerNorm blocks, residuals, weight tying; train on Tiny Shakespeare | Validation loss lands near nanoGPT’s reference run |
| C3 | BPE tokenizer | Byte-level BPE: count pairs, merge, encode, decode | Round-trips arbitrary Unicode; token counts close to tiktoken on sample text |
| C4 | Rotary position embeddings | RoPE applied to Q and K | Attention score depends only on relative offset (property test) |
| C5 | Llama-style block | Swap in RMSNorm, SwiGLU MLP and grouped-query attention | Parameter count matches the reference config |
| C6 | KV cache | Incremental decoding with a cache | Logits identical to full forward; tokens/s speedup measured |
| C7 | Mixed precision, accumulation, compile | bf16 autocast, gradient accumulation and `torch.compile` in your own loop | Table of tokens/s and peak memory per setting |
| C8 | LoRA | `LoRALinear` wrapping `nn.Linear` (A, B, alpha/r scaling, merge and unmerge) | Merged model equals adapter model outputs; trainable-parameter % printed |
| C9 | QLoRA | 4-bit NF4 base plus LoRA via bitsandbytes and PEFT | Peak VRAM report vs bf16 LoRA |
| C10 | SFT loss masking and packing | Mask prompt tokens (label -100), pack sequences, apply chat templates | Test asserts prompt labels are masked |
| C11 | DPO | The DPO loss from log-probs, by hand | Matches TRL’s `DPOTrainer` loss on a tiny batch |
| C12 | GRPO | Group-relative advantages with a verifiable reward (SQL execution) | Reward rises on a toy task |
| C13 | Tool-calling loop and resilient API client | Agent loop with typed tools, step limit, timeouts, retries with jitter | Chaos test (injected 429s and timeouts) still completes |
| C14 | Embeddings and ANN search | Brute-force cosine, then FAISS HNSW and IVF | Recall@k vs latency plot |
| C15 | Hybrid retrieval and reranking | BM25 plus dense with reciprocal rank fusion, then a cross-encoder rerank | MRR improves on your golden set |
| C16 | RAG evaluation | Golden set, hit rate, MRR, faithfulness via an LLM judge | Metrics run in CI and gate merges |
| C17 | Continuous batching and paged KV memory | Toy scheduler that admits and evicts requests each step; block table for KV | Simulation beats static batching on throughput |
| C18 | Data parallelism | Manual gradient all-reduce with `torch.distributed` on 2 processes | Gradients identical across ranks, then match `DDP` |
| C19 | ZeRO and FSDP memory math | Script for params, grads and optimizer bytes under ZeRO-1/2/3 | Within about 10% of measured peak memory |
| C20 | Tensor parallelism | Column- and row-parallel linear layers across 2 GPUs | Output equals the single-GPU layer |
| C21 | Near-duplicate removal | MinHash plus LSH on a corpus | Removal-rate report with spot-checked samples |
| C22 | Kubernetes autoscaling and probes | Load-test a deployment; tune startup and readiness probes and the HPA | Pods-vs-load graph shows scale-up and stable latency |
| C23 | Idempotent infrastructure | Terraform apply, destroy, apply for a full stack | Second apply is a no-op; plan runs in CI |

## 3. The 26-week schedule

One schedule, split into phase tables for readability. Columns: what to replicate, what to build (your mutation plus the shipped deliverable), and the concepts (C-numbers, Section 2) and cheat sheets (CS-numbers, Appendix A) to cover. Every week also includes the 2-hour Hardening block and the Section 7 checklist. **P1–P7** are the mini-projects (Section 4).

### Phase A: Foundations, PyTorch and Transformers from scratch (Weeks 1–4)

| Wk | Theme | Code-along (replicate) | Build (mutate and ship) | Concepts · Cheat sheets |
| --- | --- | --- | --- | --- |
| 1 | Dev scaffold and PyTorch core | [Karpathy Zero to Hero: micrograd](https://karpathy.ai/zero-to-hero.html); [PyTorch: Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) | `ml-template` repo with uv, ruff, pytest, pre-commit, Dockerfile and a GitHub Actions lint+test workflow. Create W&B, AWS and GCP accounts and set billing alarms | autograd and training loop by hand · CS1 |
| 2 | Training mechanics, Lightning, W&B | Zero to Hero: makemore series (bigram, MLP, BatchNorm and init, backprop); [Lightning docs](https://lightning.ai/docs/pytorch/stable/); [W&B docs](https://docs.wandb.ai/) | **P1 Training template:** port the makemore MLP to a LightningModule and DataModule; W&B logging and checkpoint Artifacts; config files; seed-reproducibility test; CI green | init, normalization, gradient flow · CS1 |
| 3 | Attention and GPT | Zero to Hero: Let’s build GPT, then Let’s build the GPT Tokenizer | Write multi-head attention and a Transformer block from scratch; train your own BPE tokenizer; train GPT on Tiny Shakespeare, then on a corpus you choose | C1–C3 · CS2, CS3 |
| 4 | Modern LLM internals and fast training | Zero to Hero: Let’s reproduce GPT-2 (124M) (bf16, `torch.compile`, flash attention, gradient accumulation, DDP; run a few hundred steps, not the full run); [Raschka: LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) or [Umar Jamil: LLaMA from scratch](https://github.com/hkproj/pytorch-llama); *(opt.)* skim the [nanochat](https://github.com/karpathy/nanochat) repo layout | **P2 MiniLLM:** GPT to Llama-style (RoPE, RMSNorm, SwiGLU, GQA) with a KV cache; unit-tested; tokens/s and memory table in W\&B; Dockerized | C4–C7 · CS4 |

### Phase B: Fine-tuning with the Hugging Face stack (Weeks 5–8)

| Wk | Theme | Code-along (replicate) | Build (mutate and ship) | Concepts · Cheat sheets |
| --- | --- | --- | --- | --- |
| 5 | Hugging Face ecosystem and evals | [HF LLM Course](https://huggingface.co/learn/llm-course) (Transformers, Datasets, Tokenizers, Hub); [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) quickstart | Fine-tune a small model with HF Trainer, then redo it in Lightning; write a standalone eval script (one harness task plus one custom metric); push to the Hub with a model card | collators, padding, chat templates · CS3 |
| 6 | LoRA and QLoRA | [Umar Jamil: LoRA](https://github.com/hkproj/pytorch-lora); [PEFT docs](https://huggingface.co/docs/peft); [Unsloth notebooks](https://github.com/unslothai/unsloth) (QLoRA) | Write your own `LoRALinear`, then swap in PEFT; QLoRA a 7–8B model on a 24 GB GPU; VRAM table (bf16 LoRA vs QLoRA) | C8–C9 · CS5 |
| 7 | Supervised fine-tuning | [smol-course](https://github.com/huggingface/smol-course) SFT units; [TRL SFTTrainer docs](https://huggingface.co/docs/trl); *(opt.)* [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) config for comparison | Build the running-domain SFT set from [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/): clean, dedupe, apply chat template, mask prompt tokens; SFT run in W\&B; held-out execution-accuracy eval | C10 · CS6 |
| 8 | Preference tuning, RL, registry | smol-course DPO unit; TRL GRPO docs; [MLflow docs](https://mlflow.org/docs/latest/) (registry and aliases) | **P3 Domain fine-tune pipeline:** QLoRA SFT, then DPO (or GRPO with a SQL-execution reward); eval report vs the base model and a hosted API model; MLflow registry with `candidate` and `champion` aliases; Hub model card; `make train` reproduces it | C11–C12 · CS6, CS8 |

### Phase C: LLM applications (Weeks 9–12)

| Wk | Theme | Code-along (replicate) | Build (mutate and ship) | Concepts · Cheat sheets |
| --- | --- | --- | --- | --- |
| 9 | LLM APIs done properly | [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) intro module; [Anthropic docs](https://docs.claude.com) and [OpenAI docs](https://platform.openai.com/docs) (structured output, tool use, streaming) | FastAPI async service: Pydantic-validated outputs, a tool-calling loop, SSE streaming, retries with backoff and timeouts, token and cost accounting, a response cache; tests run on recorded fixtures, never live calls | C13 · — |
| 10 | RAG core and vector databases | LLM Zoomcamp search and RAG modules; [LlamaIndex](https://github.com/run-llama/llama_index) starter and from-scratch RAG notebooks; [Qdrant](https://qdrant.tech/documentation/) and [pgvector](https://github.com/pgvector/pgvector) quickstarts | Idempotent ingestion pipeline (parse, chunk, embed, upsert); compare FAISS vs Qdrant vs pgvector on recall@k and latency; add BM25 hybrid search and a reranker | C14–C15 · CS7 |
| 11 | Frameworks and agents | LlamaIndex query engines and routers; [LangChain v1 release notes](https://docs.langchain.com/oss/python/releases/langchain-v1) (`create_agent`, middleware); [LangChain Academy](https://academy.langchain.com/) Intro to LangGraph | Build the same RAG three ways (raw SDK, LlamaIndex, LangChain/LangGraph); add a human-approval step on a risky tool; write a one-page ADR on when a framework earns its keep | C13 · CS7 |
| 12 | Evals and observability | [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/); [Ragas](https://github.com/explodinggradients/ragas); [MLflow GenAI docs](https://mlflow.org/docs/latest/genai/) (tracing, evaluate); *(opt.)* [Langfuse](https://langfuse.com/docs) | **P4 Production RAG service:** versioned golden set (100+ questions); offline eval as a CI gate that fails on regression; traces with retrieval context, latency and cost; prompts versioned in MLflow; Dockerized | C16 · CS8 |

### Phase D: MLOps and cloud module (Weeks 13–17)

Section 5 has the module overview, the AWS-to-GCP map and the Kubernetes GPU gotchas. This is where the P4 service becomes a deployable product.

| Wk | Theme | Code-along (replicate) | Build (mutate and ship) | Concepts · Cheat sheets |
| --- | --- | --- | --- | --- |
| 13 | Docker for ML and CI/CD | [Docker get started](https://docs.docker.com/get-started/); [TechWorld with Nana](https://www.youtube.com/@TechWorldwithNana) Docker and GitHub Actions videos; [Made With ML](https://github.com/GokuMohandas/Made-With-ML) testing and CI/CD lessons; [GitHub Actions docs](https://docs.github.com/en/actions) | Multi-stage images with uv and a BuildKit cache; compose stack (app, Qdrant, Postgres, MLflow); pipeline: lint, test, build, Trivy scan, push, eval gate | — · CS11, CS14 |
| 14 | Kubernetes core | [Kubernetes tutorials](https://kubernetes.io/docs/tutorials/) (basics, then config and stateful); [kind](https://kind.sigs.k8s.io/); [Helm docs](https://helm.sh/docs/); Nana’s Kubernetes video | Deploy P4 to a local `kind` cluster through a Helm chart: Deployment, Service, Ingress, ConfigMap and Secret, probes, requests and limits, HPA; load-test with k6 or Locust; CronJob for re-ingestion | C22 · CS12 |
| 15 | AWS | [AWS Workshops](https://workshops.aws/) (ECS/EKS, SageMaker); [SageMaker examples](https://github.com/aws/amazon-sagemaker-examples); [EKS best practices](https://aws.github.io/aws-eks-best-practices/); [Terraform tutorials](https://developer.hashicorp.com/terraform/tutorials) | Terraform VPC, ECR, S3 and IAM roles (no static keys); deploy P4 on ECS Fargate or EKS; SageMaker training job and real-time endpoint for the P3 adapter (or Bedrock); CloudWatch dashboard and alarms; cost tags | C23 · CS13 |
| 16 | GCP | [Cloud Skills Boost](https://www.cloudskillsboost.google/) labs; [vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples); [GKE samples](https://github.com/GoogleCloudPlatform/kubernetes-engine-samples) | Terraform Artifact Registry, GCS and service accounts; GitHub to GCP via Workload Identity Federation (no JSON keys); deploy P4 on Cloud Run or GKE Autopilot; Vertex AI custom training job and endpoint for the P3 adapter | C23 · CS13 |
| 17 | Platform patterns and P5 | [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) (orchestration, monitoring; self-paced); [Chip Huyen: Designing ML Systems repo](https://github.com/chiphuyen/dmls-book); *(opt.)* Airflow or Prefect scheduling re-ingestion and evals | **P5 Multi-cloud deployment:** one Terraform module set deploys P4 to AWS and GCP; CI/CD to both with canary and rollback; MLflow server behind auth; latency, error, cost and quality dashboards; runbook and cost report. Also write the capstone design doc (ADRs) | C22–C23 · CS13, CS14 |

### Phase E: Inference at scale (Weeks 18–19)

| Wk | Theme | Code-along (replicate) | Build (mutate and ship) | Concepts · Cheat sheets |
| --- | --- | --- | --- | --- |
| 18 | vLLM | [vLLM docs](https://docs.vllm.ai/) quickstart and OpenAI-compatible server; *(opt.)* [GPU MODE lectures](https://github.com/gpu-mode/lectures) on attention kernels | Serve the P3 model with LoRA adapters; benchmark TTFT, tokens/s and p95 against batch size, quantization (AWQ or FP8), prefix caching and speculative decoding; write a toy continuous-batching scheduler; benchmark report. **Capstone thread (2 h):** eval harness running against the vLLM endpoint | C17 · CS9 |
| 19 | Ray and GPU serving on Kubernetes | [Ray Data LLM](https://docs.ray.io/en/latest/data/working-with-llms.html) and the [vLLM batch-inference example](https://docs.vllm.ai/en/stable/examples/ray_serving/batch_llm_inference/); [Ray Serve](https://docs.ray.io/en/latest/serve/index.html); [KubeRay](https://github.com/ray-project/kuberay); [GKE Ray tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray) (written for TPUs, adapt to GPUs); [HyperPod and EKS blueprints](https://awslabs.github.io/ai-on-sagemaker-hyperpod/) | **P6 Inference platform:** vLLM behind Ray Serve on a GPU Kubernetes cluster (EKS or GKE), autoscaling on request load, a Ray Data LLM offline batch job, load test, cost per 1M tokens. **Capstone thread (2 h):** router between a self-hosted model and a hosted API | C17 · CS9, CS12 |

### Phase F: Distributed training and data pipelines (Weeks 20–22)

| Wk | Theme | Code-along (replicate) | Build (mutate and ship) | Concepts · Cheat sheets |
| --- | --- | --- | --- | --- |
| 20 | Multi-GPU fundamentals | [PyTorch DDP series](https://pytorch.org/tutorials/beginner/ddp_series_intro.html); [Umar Jamil: distributed training](https://github.com/hkproj/pytorch-transformer-distributed); Lightning DDP strategy docs | Manual gradient all-reduce on 2 processes, then `DDP`, then Lightning DDP via `torchrun` on 2–4 rented GPUs; profile with the PyTorch Profiler; scaling-efficiency table. **Capstone thread (2 h):** repo skeleton, Terraform modules, CI | C18 · CS10 |
| 21 | FSDP2, ZeRO and memory | [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html); [HF Accelerate](https://huggingface.co/docs/accelerate) FSDP and DeepSpeed guides; [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (memory, ZeRO and tensor-parallel chapters); TRL multi-GPU docs | Fine-tune a 7–8B model with FSDP (plus LoRA) on 4–8 GPUs; write the ZeRO memory calculator and compare it with measured peak memory; sharded checkpoint save and resume; activation-checkpointing ablation; toy tensor-parallel linear layer. **Capstone thread (2 h):** FSDP+LoRA dry run on 2 GPUs | C19–C20 · CS4, CS10 |
| 22 | Ray Train, multi-node, data pipelines | [Ray Train docs](https://docs.ray.io/en/latest/train/train.html); [datatrove](https://github.com/huggingface/datatrove) and the [FineWeb write-up](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1); *(opt.)* [CS336 Assignment 4, data filtering](https://github.com/stanford-cs336) | **P7 Distributed fine-tune with data pipeline:** Ray Data or datatrove pipeline (parse, filter, near-duplicate removal, tokenize, Parquet shards on S3 or GCS) feeding a multi-node Ray Train or FSDP job on cloud Kubernetes; kill-a-worker resume test; MFU and throughput report. **Capstone thread (2 h):** data pipeline stage 1 | C21 · CS10 |

### Phase G: Capstone (Weeks 23–26)

| Wk | Theme | Build (details in Section 4) | Concepts · Cheat sheets |
| --- | --- | --- | --- |
| 23 | Capstone I: data and training | Synthetic plus real SQL data pipeline with execution-based filtering; SFT then DPO or GRPO runs on multi-GPU cloud; every run tracked and registered | all, applied |
| 24 | Capstone II: serving and application | vLLM with LoRA on Ray Serve (EKS), schema-aware RAG, LangGraph agent loop with SQL validation, router with API fallback, read-only SQL sandbox | all, applied |
| 25 | Capstone III: quality, platform, second cloud | Eval gate in CI, tracing, dashboards and alerts, canary, secondary deployment on GKE, load test, security review | all, applied |
| 26 | Capstone IV: ship and job readiness | README and architecture diagram, ADRs, benchmark and cost reports, 3-minute demo video, blog post, tagged release; two mock ML system design interviews; resume bullets | all, applied |

**Optional stretch tracks if you are ahead of schedule:**

- Run the [nanochat](https://github.com/karpathy/nanochat) speedrun once on a rented 8×H100 node. It covers tokenizer, pretraining, SFT, eval and a chat UI in one small repo, and its leaderboard lists runs that reach GPT-2-grade capability in under about two hours on one node.
- Work through [Stanford CS336](https://cs336.stanford.edu/) Assignments 1 and 2 (BPE, Transformer, optimizer, then systems and FlashAttention in Triton). Lectures are on [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) (Spring 2025 playlist) and the assignment repos are in the [stanford-cs336 GitHub org](https://github.com/stanford-cs336).
- Use the AWS and GCP ML certification exam guides as free syllabi to find gaps. The certificates themselves are optional.

## 4. Projects

### Mini-projects (one per phase)

| # | Week | Project | You ship | Signal to a hiring manager |
| --- | --- | --- | --- | --- |
| P1 | 2 | Training template | Lightning + W&B repo with tests, config files and CI | Engineering hygiene around DL training |
| P2 | 4 | MiniLLM | From-scratch Llama-style model with tokenizer, KV cache and tests | You understand the model, not just the API |
| P3 | 8 | Domain fine-tune pipeline | QLoRA SFT plus DPO or GRPO, eval report, registry, Hub model card | End-to-end post-training with honest evals |
| P4 | 12 | Production RAG service | FastAPI, vector DB, eval gate in CI, tracing | Applied LLM engineering with quality control |
| P5 | 17 | Multi-cloud deployment | Terraform and CI/CD to AWS and GCP with canary and rollback | Real MLOps on both major clouds |
| P6 | 19 | Inference platform | vLLM behind Ray Serve on Kubernetes, load test, cost per 1M tokens | Serving and cost awareness |
| P7 | 22 | Distributed fine-tune and data pipeline | Multi-node training with fault tolerance and an MFU report | Large-scale training systems skills |

Every project gets its own repo (or a monorepo folder) with a README containing a results table, an architecture sketch and a 2-minute demo. Pin the best three on your GitHub profile.

### Capstone: QueryPilot, a production text-to-SQL analyst copilot (Weeks 23–26)

**Why this one:** it has a verifiable metric (execution accuracy), it exercises every Tier 1 tool, it plays to your data-science background (analysts are real users), and it demos well. If you prefer another domain (research-paper assistant, support copilot), keep one rule: there must be an objective eval.

**What you build, layer by layer**

- **Data:** BIRD and Spider plus synthetic question-and-SQL pairs generated by a strong hosted model and filtered by actually executing the SQL. Ray Data pipeline writes Parquet shards to S3 or GCS; datasets versioned as W&B Artifacts.
- **Training:** QLoRA or LoRA SFT, then DPO (pairs: execution-correct vs incorrect) or GRPO (execution reward) on a 7–8B open model, run with FSDP or Ray Train on multi-GPU cloud nodes. W&B tracks runs; the MLflow registry promotes `candidate` to `champion` only when the eval gate passes.
- **Serving:** vLLM with LoRA adapters behind Ray Serve on EKS (primary) and GKE (secondary, to prove portability), autoscaled on load. A router sends hard or long queries to a hosted API model, with a cost and latency policy and a circuit breaker.
- **Application:** schema-aware RAG (table and column descriptions, example queries) in Qdrant or pgvector; a LangGraph loop of retrieve schema, generate SQL, validate with sqlglot and EXPLAIN, execute in a read-only sandbox, self-repair, then answer. LlamaIndex handles ingestion and retrieval; FastAPI streams responses.
- **Quality and safety:** offline eval suite (execution accuracy, valid-SQL rate, latency, cost) as a CI gate; online tracing with MLflow or Langfuse; read-only DB role, allow-listed schemas, statement timeouts, no DDL; prompt-injection tests and PII redaction.
- **Platform:** Terraform modules for both clouds; GitHub Actions with PR checks (lint, tests, eval smoke), merge to staging (build, scan, push, deploy), and manual promotion to production with a canary; dashboards, alerts, runbook, cost report.

**Weekly milestones and acceptance criteria**

| Wk | Milestone | Acceptance criteria |
| --- | --- | --- |
| 23 | Data and training | Dataset versioned in W&B Artifacts; at least three training runs compared on the same held-out set; best checkpoint registered as `candidate` only if the eval passes |
| 24 | Serving and application | End-to-end natural-language question to answer against a sample warehouse, meeting an SLO you state up front (for example p95 under 5 s); router fallback and circuit breaker tested |
| 25 | Quality, platform, second cloud | A PR that degrades execution accuracy is blocked by CI; second cloud deployed from the same Terraform modules; canary and rollback demonstrated; load test at your SLO |
| 26 | Ship | A stranger can run a local demo from the README in 15 minutes; cost per 100 queries reported for self-hosted vs hosted API, with an honest accuracy and latency comparison |

**Definition of done:** tagged release; README with architecture diagram; at least five ADRs; benchmark and cost reports; 3-minute demo video; one technical blog post; secret scan clean. The goal of the comparison is an honest result, not a win: a smaller fine-tuned model that is cheaper but a few points less accurate is still a strong interview story if you can explain the trade-off with data.

## 5. The MLOps and cloud module (Weeks 13–17)

**Outcomes:** by the end of Week 17 you can containerize any model service, ship it through CI/CD with quality gates, run it on Kubernetes, deploy it to AWS and GCP from Terraform, and defend the cost, security and latency trade-offs. Weeks 19, 22 and 25 revisit it with GPUs and multi-node jobs.

### The three pipelines you will build

1. **Code CI (every PR):** lint, type-check, CPU unit tests on tiny fixtures, build the image, scan it.
2. **Model and prompt CI (when code, data, prompts or a model change):** smoke-train on a tiny dataset, run the offline eval against the baseline, log to W&B and MLflow, register the artifact. The merge is blocked if the metric drops beyond a tolerance.
3. **CD (on merge):** deploy to staging, run smoke and end-to-end tests, canary to production, roll back automatically on an SLO or eval regression. Cloud auth uses OIDC, so there are no long-lived keys in GitHub secrets.

### AWS and GCP side by side

Learn both in parallel so you absorb the concepts rather than one console.

| Need | AWS | GCP |
| --- | --- | --- |
| Identity for people and services | IAM roles; IRSA or EKS Pod Identity for pods | IAM service accounts; Workload Identity for GKE pods |
| CI access without stored keys | GitHub OIDC to an IAM role | Workload Identity Federation |
| Object storage | S3 | Cloud Storage (GCS) |
| Container registry | ECR | Artifact Registry |
| Serverless containers | ECS Fargate, App Runner | Cloud Run |
| Managed Kubernetes | EKS (managed node groups, Karpenter) | GKE (Autopilot or Standard) |
| Managed training and endpoints | SageMaker training jobs and endpoints; HyperPod for clusters | Vertex AI custom training and endpoints |
| Hosted foundation models | Bedrock | Vertex AI (Gemini and Model Garden) |
| Ray on Kubernetes | KubeRay on EKS | Ray Operator add-on on GKE |
| High-speed GPU networking | EFA | GPUDirect and RDMA networking on A3 and A4 VMs |
| Observability | CloudWatch, managed Prometheus and Grafana | Cloud Monitoring and Logging, managed Prometheus |
| Secrets | Secrets Manager | Secret Manager |
| Infrastructure as code | Terraform AWS provider | Terraform Google provider |
| Cost control | Budgets, tags, Spot | Budgets, labels, Spot VMs |

### Kubernetes GPU gotchas (learn them before they page you)

- Request GPUs with the `nvidia.com/gpu` resource, which needs the device plugin or GPU operator. Taint GPU nodes and add tolerations only to GPU pods.
- Use a `startupProbe` for slow model loads, and report ready only after the weights are in memory.
- Autoscale on request concurrency or queue depth, not CPU.
- Mount a memory-backed `emptyDir` at `/dev/shm`; PyTorch DataLoaders and NCCL need it.
- Cache model weights on a PVC or node-local disk instead of downloading them in every pod.
- Spot and preemptible nodes need checkpointing and graceful shutdown (`terminationGracePeriodSeconds`).
- GPU images are huge: pin digests, and pre-pull or use image streaming to cut cold starts.

### Security baseline (non-negotiable)

Roles over keys; least-privilege IAM; private networking with no public endpoints unless intended; secrets in Secrets Manager or Secret Manager (never in images or repos); scanned images; MLflow behind authentication; budget alarms on from Day 1.

### Module done criteria

- From an empty account, one command stands the stack up on each cloud and one command tears it down.
- A deliberately bad prompt change is blocked by CI.
- You can state your cost per 1M tokens for the self-hosted model and what drives it (utilization, batch size, GPU price).

## 6. Resource library

The schedule names one primary resource per week; everything else here is reference. A ✓ means the link was confirmed live in search on 11 Nov 2026. Unmarked links are long-standing canonical URLs I did not re-open. If one 404s, search the title, because repos and docs move (AWS’s distributed-training repo, for example, now lives under `awslabs`).

**Build the model yourself**

- [Karpathy: Neural Networks, Zero to Hero](https://karpathy.ai/zero-to-hero.html): micrograd through the GPT-2 reproduction, all code-along. Weeks 1–4.
- [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat) ✓: minimal training repos; nanochat spans tokenizer to chat UI and keeps a speedrun leaderboard.
- [Raschka: LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch): notebooks for GPT, Llama-style blocks and fine-tuning.
- Umar Jamil’s repos, each paired with a video: [Transformer](https://github.com/hkproj/pytorch-transformer), [LLaMA](https://github.com/hkproj/pytorch-llama), [LoRA](https://github.com/hkproj/pytorch-lora), [distributed training](https://github.com/hkproj/pytorch-transformer-distributed).
- [Stanford CS336](https://cs336.stanford.edu/) ✓ (Spring 2026 offering is live), the [stanford-cs336 assignment repos](https://github.com/stanford-cs336) ✓, and the [Spring 2025 lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) ✓.
- [PyTorch tutorials](https://pytorch.org/tutorials/) and [Lightning docs](https://lightning.ai/docs/pytorch/stable/).

**Fine-tuning and evals**

- [HF LLM Course](https://huggingface.co/learn/llm-course) and [smol-course](https://github.com/huggingface/smol-course) ✓ (SFT, then DPO, then RL, on small models you can run locally).
- [TRL](https://huggingface.co/docs/trl) ✓ (now v1), [PEFT](https://huggingface.co/docs/peft), [Accelerate](https://huggingface.co/docs/accelerate), [Unsloth](https://github.com/unslothai/unsloth), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl).
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [Ragas](https://github.com/explodinggradients/ragas), [Hamel Husain on evals](https://hamel.dev/blog/posts/evals/).
- [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/): datasets for the running domain.

**LLM applications**

- [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) ✓: free, self-paced RAG-to-application course.
- [LangChain v1 release notes](https://docs.langchain.com/oss/python/releases/langchain-v1) ✓ and [migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1) ✓; [LangChain Academy](https://academy.langchain.com/).
- [LlamaIndex](https://github.com/run-llama/llama_index); [DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses/) for fast guided labs.
- [Qdrant docs](https://qdrant.tech/documentation/), [pgvector](https://github.com/pgvector/pgvector), [FAISS](https://github.com/facebookresearch/faiss).
- [MLflow docs](https://mlflow.org/docs/latest/) ✓ (MLflow 3 GenAI), [Langfuse](https://langfuse.com/docs), [W&B docs](https://docs.wandb.ai/).

**Serving and distributed training**

- [vLLM docs](https://docs.vllm.ai/) ✓, [Ray Data LLM](https://docs.ray.io/en/latest/data/working-with-llms.html) ✓, the [vLLM plus Ray batch example](https://docs.vllm.ai/en/stable/examples/ray_serving/batch_llm_inference/) ✓, [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), [Ray Train](https://docs.ray.io/en/latest/train/train.html), [KubeRay](https://github.com/ray-project/kuberay).
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) ✓: free book on data, ZeRO, tensor, pipeline and context parallelism with benchmarks and a memory predictor.
- [PyTorch DDP series](https://pytorch.org/tutorials/beginner/ddp_series_intro.html), [FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [Stas Bekman: ML Engineering](https://github.com/stas00/ml-engineering) (practical GPU-cluster notes).
- [datatrove](https://github.com/huggingface/datatrove) and the [FineWeb write-up](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) for large-scale data pipelines.

**MLOps and cloud**

- [Docker get started](https://docs.docker.com/get-started/), [Kubernetes tutorials](https://kubernetes.io/docs/tutorials/), [kind](https://kind.sigs.k8s.io/), [Helm](https://helm.sh/docs/), [GitHub Actions](https://docs.github.com/en/actions), [Terraform tutorials](https://developer.hashicorp.com/terraform/tutorials), and [TechWorld with Nana](https://www.youtube.com/@TechWorldwithNana) for video walkthroughs of each.
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML), [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) ✓ (no live 2026 cohort, but the whole course is available self-paced), [Chip Huyen: Designing ML Systems repo](https://github.com/chiphuyen/dmls-book).
- **AWS:** [awslabs/awsome-distributed-training](https://github.com/awslabs/awsome-distributed-training) ✓ (reference architectures and tests for HyperPod, ParallelCluster, Batch and EKS, including an FSDP test case), [HyperPod and EKS blueprints](https://awslabs.github.io/ai-on-sagemaker-hyperpod/) ✓, [SageMaker examples](https://github.com/aws/amazon-sagemaker-examples), [AWS Workshops](https://workshops.aws/), [EKS best practices](https://aws.github.io/aws-eks-best-practices/).
- **GCP:** [GKE samples](https://github.com/GoogleCloudPlatform/kubernetes-engine-samples) ✓, [AI Hypercomputer tutorials](https://docs.cloud.google.com/ai-hypercomputer/docs/tutorials) ✓ (vLLM on GKE, fine-tuning with FSDP, multi-node training), the [GKE Ray and vLLM tutorial](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray) ✓ (TPU version), [vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples), [Cloud Skills Boost](https://www.cloudskillsboost.google/).

## 7. Weekly code-review checklists

### Every week (paste into your PR template)

- **Code quality:** typed code; ruff and mypy (or pyright) clean; no logic that lives only in a notebook; functions small enough to test.
- **Tests:** shape, dtype and device tests; a seeded determinism test; an overfit-one-batch test; the whole suite runs on CPU in CI in under 5 minutes.
- **Config:** no hard-coded paths or hyperparameters; config files (Hydra or pydantic-settings); secrets only from the environment or a secret manager.
- **Reproducibility:** pinned lockfile (`uv.lock`); seeds set; data version, config and git SHA logged with every run; one command reproduces the headline result (`make train`).
- **Tracking:** every run logs params, metrics, environment and artifacts to W&B or MLflow; failed runs are tagged, not deleted.
- **Docker:** the image builds in CI; non-root user; `.dockerignore`; pinned base image; no secrets in layers; size and build time noted.
- **CI:** lint, test and build run on every PR; main is protected; caches are on; a status badge is in the README.
- **Observability:** structured logs with request IDs; latency, token and cost metrics for anything served.
- **Cost and safety:** teardown verified; budget alarms on; least-privilege IAM; nothing public without auth.
- **Docs:** README says what, why and how to run, with a results table; one ADR for this week’s key decision; a screenshot or GIF.

### Add-on items by week

| Wk | Also verify |
| --- | --- |
| 1 | The template repo clones and passes CI from scratch in under 10 minutes; pre-commit blocks a bad commit (demo it); billing-alarm screenshot saved in docs |
| 2 | Same seed gives the same loss curve within tolerance; the LightningModule holds no data logic (the DataModule owns it); the W&B run has config, git SHA and a checkpoint Artifact |
| 3 | Your attention matches `F.scaled_dot_product_attention`; a test proves the causal mask leaks no future tokens; the tokenizer round-trips arbitrary Unicode |
| 4 | KV-cache logits equal the full forward pass; a throughput table (tokens/s, memory) covers autocast, compile and accumulation; dtypes are explicit, with no silent mixing |
| 5 | Collator, padding and truncation are tested; the eval script is deterministic and separate from training; the model card lists data, metrics and limitations |
| 6 | LoRA merge and unmerge equality test passes; an assert shows only adapter params require grad; VRAM recorded for bf16 LoRA vs QLoRA |
| 7 | Prompt tokens are masked in the loss (tested); the train and validation split cannot leak (split by database for SQL); the dataset is versioned |
| 8 | Base, SFT and DPO are compared on one held-out set with several seeds or confidence intervals; registry promotion requires an eval pass; `make train` reproduces within tolerance |
| 9 | Timeouts, retries with jitter and idempotency; no API keys in code or logs; tests use recorded fixtures, not live calls; outputs are schema-validated with a repair or fallback path |
| 10 | Chunking choice is justified by a metric; a recall@k and latency table covers at least three stores; re-running ingestion does not duplicate records |
| 11 | All three implementations share one eval set; the ADR records trade-offs; tools have typed inputs, timeouts and approval gates for risky actions |
| 12 | Golden set is versioned and has 100+ items; CI fails on a regression (prove it with a deliberately bad prompt); traces include retrieved context, latency and cost |
| 13 | Image size and build time are tracked; the vulnerability scan has no critical findings; a CI cache hit is demonstrated; the pipeline gates on tests and eval |
| 14 | Probes suit slow model loads; requests and limits are set; the HPA scales under a load test; no secrets in manifests; `helm lint` runs in CI |
| 15 | IAM has no wildcard admin; no static access keys; `terraform plan` runs in CI with remote locked state; cost tags and a budget alarm exist; lab endpoints are torn down |
| 16 | No JSON service-account keys (Workload Identity Federation instead); remote Terraform state; budget alert set; services are non-public unless intended; teardown verified |
| 17 | One module set deploys to both clouds; canary and rollback demonstrated; MLflow is behind auth; dashboards show latency, errors, cost and quality; a runbook exists |
| 18 | Benchmark report states hardware, vLLM version, flags and prompt and output length distribution; p50, p95 and p99 latency and TTFT, not only throughput; quality delta of quantized vs bf16 measured |
| 19 | Autoscaling tested on a load metric; graceful drain verified; cost per 1M tokens stated with utilization; idle GPU nodes scale down |
| 20 | Only rank 0 logs and checkpoints; per-rank seeding handled; scaling-efficiency table for 1, 2 and 4 GPUs; no hard-coded world size |
| 21 | Sharded checkpoint save then resume gives the same loss trajectory; memory estimate vs measured recorded; activation-checkpointing trade-off recorded; NCCL timeouts documented |
| 22 | Pipeline is idempotent and resumable; the worker-kill test passes; shards are versioned and checksummed; MFU report names the bottleneck |
| 23 | Runs registered with lineage (data version, config, model); eval gates promotion; synthetic data filtered by execution; cost tracked per run |
| 24 | Read-only DB role and statement timeout enforced; prompt-injection tests pass; router fallback and circuit breaker tested; load-test report written |
| 25 | Both clouds deploy through CI; dashboards and alerts exist; a rollback drill is documented; IAM, network and secrets reviewed |
| 26 | A stranger can run the demo from the README in 15 minutes; ADRs, benchmark, cost report, demo video and tagged release exist; secret scan (for example gitleaks) is clean |

## Appendix A: One-page cheat sheets

Read the matching sheet for 15 minutes before each code-along, then write your own five-line version in your repo’s `notes/` folder. Each sheet below is the compact core; expand it to a full page with your own examples as you go.

### CS1. Autograd and the training loop

- **Mental model:** the forward pass records a graph; `loss.backward()` fills `.grad` on every parameter; the optimizer step consumes those gradients.
- **Code hook:** `zero_grad`, forward, loss, `backward`, clip, `step`, scheduler step.
- **Gotchas:** forgetting `zero_grad`; validating without `model.eval()` and `torch.no_grad()`; calling `.item()` every step (forces a GPU sync); in-place ops on tensors autograd still needs.
- **Numbers:** initial cross-entropy should be about ln(vocab size) (ln 50,257 ≈ 10.8); if it isn’t, init or labels are off. Prove any loop by overfitting one batch.

### CS2. Attention and Transformers

- **Mental model:** each token makes a query, a key and a value; scores = QKᵀ/√d; softmax; weighted sum of values. A causal mask blocks future tokens. Multi-head means several smaller attentions in parallel, concatenated and projected.
- **Block:** x + Attn(Norm(x)), then x + MLP(Norm(x)).
- **Gotchas:** wrong mask shape or dtype; missing 1/√d scale; dropout left on at eval; mixing up (batch, heads, seq, dim) axes.
- **Numbers:** attention cost grows with sequence length squared; a standard block has about 12·d² parameters, mostly MLP and projections; use `F.scaled_dot_product_attention` for the fused kernel.

### CS3. Tokenization

- **Mental model:** BPE starts from bytes and repeatedly merges the most frequent adjacent pair; the vocabulary is the base bytes plus the merges.
- **Code hook:** count pairs, merge, repeat; encoding replays the merges in order.
- **Gotchas:** pre-tokenization regex and whitespace; special tokens; chat templates change the token stream (use `apply_chat_template`); a tokenizer and model mismatch fails silently; non-Latin scripts cost more tokens per word.

### CS4. Mixed precision and memory math

- **Mental model:** training memory = weights + gradients + optimizer states + activations.
- **Numbers:** classic mixed-precision AdamW costs about 16 bytes per parameter (bf16 weights and grads, an fp32 master copy, two fp32 Adam moments), so a 7B model needs over 100 GB before activations. Inference needs parameters × bytes per parameter (7B in bf16 is about 14 GB) plus the KV cache.
- **Precision:** bf16 has fp32’s exponent range and needs no loss scaling; fp16 needs a GradScaler.
- **Levers:** gradient accumulation (bigger effective batch, same memory); activation checkpointing (roughly a third more compute for far less activation memory); LoRA and QLoRA (shrink gradients and optimizer state); sharding (FSDP, ZeRO).

### CS5. LoRA and QLoRA

- **Mental model:** freeze W and learn a low-rank update ΔW = (α/r)·B·A, with B initialized to zero so training starts exactly at the base model.
- **Numbers:** trainable parameters per adapted layer are about r·(d\_in + d\_out). Merge W + ΔW for zero-overhead serving.
- **QLoRA:** base weights stored in 4-bit NF4 (with double quantization), compute in bf16, adapters in bf16, paged optimizers to absorb memory spikes.
- **Knobs:** rank 8–64; alpha often r or 2r; target all linear layers; learning rate around 1e-4 to 2e-4.
- **Gotchas:** chat-template mismatch between training and serving; merging adapters into a quantized base is lossy; forgetting to prepare the k-bit model when using gradient checkpointing.

### CS6. SFT, DPO, GRPO

- **SFT:** next-token loss on assistant tokens only; teaches format and baseline behavior. Start here.
- **DPO:** learns from (prompt, chosen, rejected) triples; loss = −log σ(β\[(log π(chosen) − log π\_ref(chosen)) − (log π(rejected) − log π\_ref(rejected))\]). It needs a reference model (with LoRA, the adapter-disabled model can serve).
- **GRPO:** sample G completions per prompt, score each with a verifiable reward (unit tests, SQL execution), advantage = (reward − group mean) / group std, policy-gradient update with a KL penalty, no value model.
- **Gotchas:** reward hacking (read the samples); a β that is too low drifts; DPO length bias; tiny preference sets overfit. TRL v1 ships SFT, DPO, GRPO, RLOO and reward modeling as its stable trainers.

### CS7. RAG and vector search

- **Pipeline:** parse, chunk, embed, index, retrieve (hybrid), rerank, prompt, generate, cite.
- **Retrieval:** dense = cosine similarity on embeddings; sparse = BM25; fuse with reciprocal rank fusion, score = Σ 1/(k + rank), k around 60.
- **ANN indexes:** HNSW (graph, fast, memory-heavy) vs IVF-PQ (clustered and compressed, smaller, lower recall).
- **Chunking:** roughly 200–800 tokens with overlap and metadata; keep document structure.
- **Failure modes:** bad chunking, no metadata filters, stale index, retrieval never measured. Measure retrieval (recall@k, MRR) separately from generation (faithfulness, relevance).

### CS8. LLM evals

- **Three layers:** deterministic checks (schema valid, SQL executes); reference metrics (execution accuracy, exact match, pass@k); LLM-as-judge with a rubric (calibrate against human labels, prefer pairwise).
- **Start with error analysis:** read 50 real traces before choosing metrics.
- **Golden set:** 100+ representative cases, sliced by failure type, run in CI, regressions tracked.
- **Gotchas:** judge biases (position, verbosity, self-preference); test-set contamination; single-run noise (use seeds or confidence intervals); evaluating on training data.

### CS9. vLLM serving

- **Why it’s fast:** continuous batching (requests join and leave at every decode step), PagedAttention (KV cache in fixed blocks, little fragmentation), prefix caching, chunked prefill.
- **Measure:** time to first token (TTFT), inter-token latency, tokens/s, p95 and p99. Batching up raises throughput and also latency.
- **Numbers:** KV cache per token ≈ 2 × layers × KV heads × head dim × bytes per element; this, more than the weights, often caps concurrency.
- **Knobs:** `--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--quantization`.
- **Gotchas:** OOM from an oversized max model length; cold starts from weight downloads; chat-template mismatch; benchmarking with unrealistic prompt lengths.

### CS10. Distributed training

- **DDP:** replicate the model, shard the data, all-reduce gradients (overlapped with backward).
- **FSDP and ZeRO:** shard optimizer states (ZeRO-1), plus gradients (ZeRO-2), plus parameters (ZeRO-3 and FSDP), gathering parameters just in time.
- **Tensor parallel:** split matmuls across GPUs; needs fast intra-node links. **Pipeline parallel:** split layers into stages; watch the bubble. **Context parallel:** for very long sequences.
- **Rule of thumb:** fits on one GPU, use DDP; doesn’t fit, use FSDP or ZeRO-3; very large, add tensor parallel inside a node and pipeline across nodes.
- **Track:** MFU, tokens/s/GPU, scaling efficiency. **Gotchas:** NCCL timeouts; uneven batch counts that hang ranks; logging from every rank; non-deterministic dataloader seeds; sharded vs full checkpoint formats.

### CS11. Docker for ML

- **Layer caching:** copy dependency files first, install, then copy source. Multi-stage: build in a fat image, ship a slim runtime.
- **Basics:** use uv for installs, pin base image digests, match driver, CUDA and PyTorch versions, run with the NVIDIA Container Toolkit (`--gpus all`).
- **Weights:** don’t bake large weights into images; mount a volume or pull into a cache at startup.
- **Hygiene:** run as non-root; `.dockerignore` data and checkpoints.
- **Gotchas:** bloated images from duplicate CUDA libraries; too-small `/dev/shm` (`--shm-size`); secrets in build args.

### CS12. Kubernetes for ML

- **Objects:** Pod, Deployment (stateless replicas), Service (stable address), Ingress (HTTP in), ConfigMap and Secret, PVC (storage), Job and CronJob (batch), HPA (autoscale).
- **GPUs:** request `nvidia.com/gpu`, taint GPU nodes, tolerate in GPU pods, select by GPU type.
- **Probes and scaling:** `startupProbe` for slow loads; ready only after the model is loaded; autoscale on concurrency or queue depth.
- **Training:** Jobs plus an operator (Kubeflow Training Operator, KubeRay).
- **Gotchas:** OOMKilled from tight limits; huge image pulls; missing `/dev/shm`; spot-node evictions.

### CS13. Cloud mapping and cost

- **Order of operations:** identity (roles, not keys), network (private by default), storage (same region as compute to avoid egress), then compute.
- **Purchasing:** spot or preemptible for training with checkpoints; on-demand baseline for serving; scale to zero where you can.
- **Numbers:** cost per token = GPU dollars per hour ÷ (tokens/s × 3600); utilization is the lever.
- **Gotchas:** forgotten GPU instances; NAT and egress charges; orphaned disks; managed endpoints that bill while idle. The AWS and GCP side-by-side table is in Section 5.

### CS14. CI/CD for ML

- **Three loops:** code CI (lint, types, tests, build); model and prompt CI (smoke-train, eval vs baseline, register); CD (staging, smoke tests, canary, auto-rollback).
- **Triggers and gates:** a code, data, prompt or model change triggers it; tests pass, eval at or above baseline minus tolerance, image scan clean, `terraform plan` reviewed.
- **Auth:** OIDC to the clouds, never long-lived keys.
- **Gotchas:** slow CI (cache dependencies and models); flaky GPU tests (default to tiny CPU tests); eval nondeterminism (fix seeds, use temperature 0, allow tolerances).

## Appendix B: Triage rules if you fall behind

1. Drop the *(opt.)* items first.
2. Shorten Phase A before Phase D or F. Watch the GPT-2 reproduction at 1.5× and code only the key parts.
3. Protect P3, P4, P5 and P6; they map most directly to job descriptions. Shrink P1, P2 and P7 if you must.
4. Never skip Hardening or the weekly checklist.
5. If compute budget is the constraint, run P7 on 2 GPUs and keep the memory calculator and the failure-recovery test.
