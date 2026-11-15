# Week 3: Attention, GPT and Tokenizers from Scratch

*Week 3 of the 26-week plan (Phase A: Foundations). Covers concepts C1–C3 and cheat sheets CS2–CS3 from the [main plan](https://claude.ai/artifact/FD56r2oaE9BXX8VZPiQAeo). 15 hours: 1 cheat sheet, 5 code-along, 6 build, 2 hardening, 1 review. Links marked ✓ were confirmed live in search on 15 Nov 2026; unmarked links are long-standing canonical URLs I did not re-open, so search the title if one has moved.*

## 1. What you will have at the end of the week

1. A causal multi-head attention module written from a blank file, with automated tests proving it matches PyTorch’s built-in attention and cannot see future tokens.
2. A GPT model (embeddings, pre-LayerNorm Transformer blocks, tied output head, generation with temperature and top-k) with a parameter-count audit.
3. A byte-level BPE tokenizer you trained yourself, with regex pre-tokenization, special tokens, save and load, and a property-test suite.
4. Trained models: the video’s character-level GPT, a GPT on your own BPE tokenizer compared fairly on a tokenizer-independent metric, and one run on a corpus you chose.
5. A repo that passes CI, logs to W&B, builds in Docker, and has a README results table and one ADR.
6. Five-line notes on CS2 and CS3 in your own words.

This file contains no solution code. Every build task is written as an exercise with acceptance criteria.

## 2. Everything to cover this week

| Area | You can explain and implement | Hours |
| --- | --- | --- |
| Language-modeling setup | Next-token prediction, context window (block size), batches, train and validation split, why loss at initialization should sit near ln(vocab size) | 2, 11 |
| Self-attention | Queries, keys and values; scaled dot-product scores; why scale; softmax; weighted aggregation as a matrix multiply; causal masking | 3, 7 |
| Multi-head attention | Splitting channels into heads, computing all heads in one batched operation, output projection | 3, 7 |
| Transformer block | Pre-LayerNorm, residual connections, MLP with 4x expansion, dropout, what each piece does for trainability | 4, 8 |
| Positional information | Learned positional embeddings (RoPE arrives in Week 4); attention alone ignores token order | 3, 8 |
| GPT model | Token and position embeddings, stack of blocks, final norm, output head, weight tying, parameter counting | 4, 8 |
| Generation | Autoregressive sampling, cropping to the context window, temperature, top-k | 8 |
| Unicode and bytes | Code points vs UTF-8 bytes; why byte-level BPE has no unknown tokens | 5, 9 |
| Byte-pair encoding | Pair counting, merges, vocabulary construction, encoding in learned merge order, decoding | 5, 9 |
| Pre-tokenization and special tokens | Regex splitting by character category, GPT-2 vs GPT-4 differences, end-of-text, safe handling of special strings in user text | 6, 10 |
| Vocabulary-size trade-offs | Compression ratio, embedding-table size, sequence length, effective context | 9–11 |
| Tokenizer-caused LLM quirks | Spelling, arithmetic, whitespace, non-English text, code indentation | 6, 10 |
| Evaluation | Validation loss, bits per character for cross-tokenizer comparison, reading samples critically | 11–12 |
| Engineering | Property tests, seeded determinism, overfit-one-batch, CI, experiment tracking, Docker | 13–14 |

**Out of scope this week** (avoid the rabbit holes): RoPE, RMSNorm, SwiGLU, grouped-query attention and KV cache (Week 4); mixed precision, `torch.compile`, flash-attention internals and DDP (Week 4 onward); encoder-decoder and cross-attention (watch the video’s five-minute note, nothing more); fine-tuning and RLHF (the video’s closing chapter, at 1.5× speed).

## 3. Resources for Week 3

### Primary (do these)

| Resource | What it is | Time | Hours |
| --- | --- | --- | --- |
| [Let’s build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) ✓ | Karpathy’s code-along: a character-level GPT on Tiny Shakespeare. The chapter list is in the video description | 1 h 56 min | 2–4 |
| [ng-video-lecture repo](https://github.com/karpathy/ng-video-lecture) ✓ and the [video’s Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) ✓ | Reference code for the video. Use it to unblock yourself or to compare afterwards, not to copy | reference | 2–4, 8 |
| [Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) ✓ and the [video’s Colab](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing) ✓ | Builds byte-level BPE, then regex splitting, special tokens, tiktoken, SentencePiece, and why many LLM quirks trace back to tokenization | just over 2 h | 5–6 |
| [minbpe repo](https://github.com/karpathy/minbpe) ✓, [exercise.md](https://github.com/karpathy/minbpe/blob/master/exercise.md) ✓, [lecture.md](https://github.com/karpathy/minbpe/blob/master/lecture.md) ✓ | Clean reference tokenizer plus a step-by-step build-your-own progression. Attempt each step yourself before reading the solution | about 1 h | 6, 9, 10 |
| [3Blue1Brown: Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) ✓ and its [text version](https://www.3blue1brown.com/lessons/attention) ✓ | Visual intuition for queries, keys, values, masking and multiple heads | 26 min | 1 |
| [Tiktokenizer](https://tiktokenizer.vercel.app) ✓ | In-browser playground showing how GPT-style tokenizers split text | 15 min | 6, 10 |
| [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | The roughly 1 MB text used by the video | n/a | 2, 11 |
| [Main plan, Appendix A](https://claude.ai/artifact/FD56r2oaE9BXX8VZPiQAeo): CS2 and CS3 | The two one-page cheat sheets | 30 min | 1 |

### Reading and reference (secondary)

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): read the abstract and Section 3 (attention, multi-head, feed-forward, positional encoding) after Hour 3. Skip the translation experiments.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): decoder-only, masked self-attention, generation. Read during the Hour 4 training run.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): the original encoder-decoder picture; optional in Hour 15.
- [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) ✓: Section 2.2 explains byte-level BPE and the category-splitting rule.
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): the paper that brought BPE to NLP; abstract and algorithm section only.
- [Layer Normalization](https://arxiv.org/abs/1607.06450), [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (pre-LN vs post-LN; abstract and conclusion) and [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (abstract and Figure 2).
- Sebastian Raschka’s LLMs-from-scratch: [ch02 text data](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02), [ch03 attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03), [ch04 GPT model](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04). A second implementation to read after you have written your own.
- [nanoGPT](https://github.com/karpathy/nanoGPT) ✓: read its model file only after finishing Exercise E3.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/): optional code reading.
- [HF LLM Course: Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5): an alternate explanation if BPE does not click in Hour 5.
- [tiktoken](https://github.com/openai/tiktoken) ✓ and [SentencePiece](https://github.com/google/sentencepiece) ✓: reference implementations for Hour 10 and the stretch list.
- Tooling docs: [PyTorch scaled\_dot\_product\_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [pytest](https://docs.pytest.org/), [Hypothesis](https://hypothesis.readthedocs.io/) (property-based tests), [W&B](https://docs.wandb.ai/), [ruff](https://docs.astral.sh/ruff/), [uv](https://docs.astral.sh/uv/).
- Corpus ideas for Hour 12: [Project Gutenberg](https://www.gutenberg.org/), [Spider](https://yale-lily.github.io/spider) (the running SQL domain), or your own code and documents.

## 4. Hour-by-hour plan

Hours can be split across days; a suggested calendar is in Section 7. Put at least one night between the code-along hours (2–6) and the build hours (7 onward), because the build is a rebuild from memory and works best after sleep.

| Hour | Type | Focus | Output |
| --- | --- | --- | --- |
| 1 | Cheat sheet | Mental model for attention and tokenization | Written predictions and notes |
| 2 | Code-along | GPT video I: data, bigram baseline, the matrix-multiply trick | Bigram model running |
| 3 | Code-along | GPT video II: self-attention, single and multi-head | Working attention heads |
| 4 | Code-along | GPT video III: block, residual, LayerNorm, scale-up | Trained character-level GPT |
| 5 | Code-along | Tokenizer video I: Unicode and BPE train, encode, decode | Naive BPE working |
| 6 | Code-along | Tokenizer video II: regex, special tokens, tiktoken, SentencePiece | Exercise plan for E6–E9 |
| 7 | Build | E1–E2: causal multi-head attention and its tests | Module plus passing tests |
| 8 | Build | E3–E5: GPT model, parameter audit, sampling | Model, audit table, samples |
| 9 | Build | E6–E7: BPE trainer, encoder, decoder, tests | Tokenizer v1 plus tests |
| 10 | Build | E8–E9: regex, special tokens, persistence, behavior lab | Tokenizer v2 plus lab notes |
| 11 | Build | E10: character-level vs BPE, compared fairly | Comparison table |
| 12 | Build | E11: GPT on a corpus you chose | Trained model plus sample review |
| 13 | Hardening | E12: test suite, types, lint, CI | Green CI |
| 14 | Hardening | E13: W&B, Docker, README, ADR | Reproducible repo |
| 15 | Review | Checklist, notes, self-check, tag release | Tag `week-03` |

### Hour 1 · Cheat sheet · Mental model before code

- **Focus:** know what you are about to build so the video is confirmation, not confusion.
- **Materials:** CS2 and CS3 in the main plan (15 min each); 3Blue1Brown attention video (26 min). If Transformers are completely new, watch his preceding chapter, [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) ✓, first and drop the second cheat sheet to Hour 15.
- **Do:** read both sheets, watch the video once without pausing, then write your predictions (below) on paper before looking anything up.
- **Predictions to write down:** (1) What does a causal mask do to the matrix of attention scores? (2) Why divide scores by the square root of the head size? (3) If you shuffle the input tokens and remove both the mask and positional information, what happens to the outputs? (4) Why do production tokenizers work on bytes instead of characters?
- **Output:** your four predictions, plus a half-page outline of the week in your own words. You will check the predictions in Hours 3 and 5.

### Hour 2 · Code-along · GPT video, first third

- **Focus:** the language-modeling setup, and the one trick that makes attention work: aggregating the past with a lower-triangular matrix multiply.
- **Materials:** the GPT video from the start through the matrix-multiply and softmax versions of averaging (chapter “the trick in self-attention: matrix multiply as weighted aggregation”); the repo and Colab as unblockers; Tiny Shakespeare.
- **Do:** spend the first 10 minutes on setup: confirm a GPU runtime, clone your Week 1 `ml-template`, create a `week-03` branch. Then type along. Pause before each new idea and predict the output shape or value. Keep your code in files inside your repo, not only in Colab.
- **Checkpoint questions (answer in your notes):** What are block size and batch size, and what does one training example look like? Why does a bigram model’s loss at initialization sit near ln(vocab size)? Why does multiplying by a row-normalized lower-triangular matrix give a running average of the past? How does masking with negative infinity before softmax produce the same weights?
- **Output:** a bigram model that trains and generates (nonsense is expected), and notes on the three versions of aggregation.

### Hour 3 · Code-along · GPT video, attention

- **Focus:** self-attention, then multiple heads.
- **Materials:** the video’s chapters on positional encoding, “THE CRUX OF THE VIDEO: version 4: self-attention”, the numbered notes that follow it (especially why attention is communication and why we scale), inserting a single attention block, and multi-headed self-attention. Return to the 3Blue1Brown text lesson whenever the query, key and value notation blurs.
- **Do:** type along. After the crux section, close the video and redraw from memory the tensor shape at each step of one head: input, queries, keys, values, scores, masked scores, weights, output.
- **Checkpoint questions:** check your Hour 1 predictions 1 and 2 and correct them in writing. What happens to the softmax at initialization without scaling? Why does attention need positional information? Why is there no communication across the batch dimension?
- **Output:** a working single head and a multi-head version (the video’s version concatenates separate heads), with loss clearly below the bigram baseline.

### Hour 4 · Code-along · GPT video, block and scale-up

- **Focus:** assemble a Transformer block, then scale it up and train.
- **Materials:** the video’s chapters on feed-forward layers, residual connections, LayerNorm (and its relation to the BatchNorm you saw in makemore), scaling up with dropout, the quick nanoGPT walkthrough (it previews the batched multi-head you build in E1; watch at 1.25×), and the closing ChatGPT discussion (1.5×). While the model trains, read [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/).
- **Do:** type along, launch the scaled-up run on a GPU, read while it trains, then save the checkpoint and generate three samples.
- **Mutate #1 (about 15 minutes):** rerun a short experiment changing exactly one thing (number of heads, block size, or dropout). Write what you expected and what happened.
- **Output:** a trained character-level GPT with validation loss in the neighborhood of 1.5 (the video ends near there), samples saved under `notes/`, and the mutate note.

### Hour 5 · Code-along · Tokenizer video, first part

- **Focus:** bytes, Unicode and the BPE algorithm.
- **Materials:** the tokenizer video from the start through your first BPE training, encode and decode (stop before the regex-splitting chapter; use the chapter list in the description); the video’s Colab; optionally minbpe’s lecture.md as a text version.
- **Do:** watch at 1.25×. Type along only the BPE core (pair counting, merging, training loop, encode, decode) and skim the rest. For three strings (plain ASCII, accented Latin text, an emoji), write the number of code points and the number of UTF-8 bytes, and note why the difference matters.
- **Checkpoint questions:** Why start from 256 byte tokens? What does merge order mean at encode time? What goes wrong when you decode an arbitrary sequence of token ids? Check your Hour 1 prediction 4.
- **Output:** a naive BPE that trains, encodes and decodes a paragraph, plus notes.

### Hour 6 · Code-along · Tokenizer video, what real tokenizers add

- **Focus:** regex pre-tokenization, special tokens, comparison with production tokenizers.
- **Materials:** the rest of the video: regex splitting (GPT-2 vs GPT-4 patterns), tiktoken, the GPT-2 encoder walkthrough, special tokens (chapter at 1:18:26), the “minbpe exercise time” chapter (1:25:28), SentencePiece (1:28:42), then vocabulary-size considerations and the tokenization quirks at the end (watch these at 1.5×). Then read minbpe’s exercise.md and the README sections on the regex and GPT-4 tokenizers.
- **Do:** watch, then write a plan for E6–E9 using the exercise.md steps. In Tiktokenizer, paste three strings of your choice (for example a number with many digits, indented code, a non-English sentence) and screenshot the splits.
- **Output:** an annotated exercise plan, and a list of five tokenization quirks you will investigate in E9.

### Hour 7 · Build · E1–E2: causal multi-head attention, then prove it

- **Focus:** rebuild attention from a blank file and verify it against a trusted implementation.
- **Materials:** your Hour 3 shape diagram (not your Hour 3 code), the [PyTorch scaled\_dot\_product\_attention docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), the 3Blue1Brown text lesson. Open the video’s nanoGPT walkthrough chapter only if you have been stuck for more than 20 minutes.
- **Rule:** tutorial closed. This is the “rebuild” step of Replicate, Mutate, Rebuild.

**E1. Causal multi-head attention module (about 35 minutes)**

1. The module takes a batch of token embeddings shaped (batch, time, channels) and returns the same shape.
2. Constructor inputs: channels, number of heads, maximum context length, dropout probability. Derive head size from channels and heads, and fail loudly if channels are not divisible by heads.
3. Compute queries, keys and values for all heads in one batched projection each, then reshape into heads. No Python loop over heads in the forward pass.
4. Scores are scaled dot products between queries and keys. Apply a causal mask so position t can only attend to positions up to t, then softmax, then dropout on the weights, then take the weighted sum of values.
5. Merge the heads back into channels and apply an output projection and residual dropout.
6. Build the mask once and store it as a non-trainable buffer so it follows the module across devices and is excluded from the optimizer and the parameter count.
7. Provide an opt-in debug path that returns the attention weights, used by the tests below.

**E2. Verification suite (about 25 minutes)**

1. **Equivalence:** with identical projected queries, keys and values, your attention output matches PyTorch’s built-in scaled dot-product attention with its causal option, within 1e-5 in fp32. Cover at least three (heads, channels, time) configurations.
2. **No future leakage:** change every token after position t and show that outputs at positions up to t are bit-for-bit identical.
3. **Weights are a valid distribution:** each row sums to one, and every entry above the diagonal is exactly zero.
4. **Batch independence:** changing one sample in the batch leaves every other sample’s output unchanged.
5. **Shape contract:** input and output shapes match for several batch sizes and context lengths up to the maximum, and a clear error is raised beyond it.

**Done when:** all tests pass on CPU, and you can explain every line of E1 to someone without looking at notes.

### Hour 8 · Build · E3–E5: the GPT model, a parameter audit, sampling

- **Focus:** assemble blocks into a GPT you can count, train and sample from.
- **Materials:** your Hour 4 notes; Section 3 of the Attention paper; [Raschka ch04](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) and [nanoGPT](https://github.com/karpathy/nanoGPT) only after you finish E3, for comparison.

**E3. Transformer block and GPT (about 30 minutes)**

1. A block applies normalization before attention and before the MLP (pre-LayerNorm), each followed by a residual add. The MLP expands width by 4x, applies a non-linearity (ReLU in the video, GELU in nanoGPT; try both and note any difference), and projects back.
2. The model has token embeddings, learned position embeddings, N blocks, a final normalization and an output head projecting to the vocabulary.
3. Add an option to tie the output head’s weights to the token embedding.
4. Keep all hyperparameters (layers, heads, width, context length, dropout, vocabulary size) in one config object that is saved with every checkpoint.
5. The forward pass returns logits, and also the cross-entropy loss when targets are supplied.
6. Generation is an autoregressive loop that crops the running context to the maximum context length.

**E4. Parameter audit (about 15 minutes)**

1. Before counting, predict the parameter count of three configurations (tiny, the video’s size, a wider one) using the rule of thumb of about 12 × layers × width² for the blocks, plus embeddings.
2. Count the real parameters, tabulate prediction vs actual, and explain each gap (biases, normalization parameters, position embeddings, tied vs untied head).
3. Assert that the loss at initialization is within a tolerance of ln(vocabulary size).

**E5. Sampling (about 15 minutes)**

1. Add temperature and top-k to generation, with a seed so output is reproducible.
2. Generate from your Hour 4 checkpoint at three settings and write two sentences on how diversity and coherence change.

**Done when:** the model overfits a single small batch to near-zero loss, the audit table is in your notes, and sampling is reproducible under a fixed seed.

### Hour 9 · Build · E6–E7: your own BPE tokenizer

- **Focus:** train, encode and decode with byte-level BPE, and test it hard.
- **Materials:** your Hour 5 notes; step 1 and step 2 of minbpe’s [exercise.md](https://github.com/karpathy/minbpe/blob/master/exercise.md); optionally the HF course page on BPE. Attempt each step before reading any hint, and close minbpe’s source before you write.

**E6. BPE trainer, encoder, decoder (about 40 minutes)**

1. Start from text converted to UTF-8 bytes, so the base vocabulary is the 256 byte values.
2. Training repeatedly finds the most frequent adjacent pair of tokens, assigns it a new id and replaces it everywhere, until the target vocabulary size is reached. Break ties deterministically and record merges in the order learned.
3. Encoding converts text to bytes and applies merges in the order they were learned (earliest merge first), not by frequency in the new text.
4. Decoding turns ids back into bytes and then into text, handling invalid UTF-8 without crashing.
5. Before running anything, do the first three merges by hand on a short repetitive string, then check the code agrees. The small worked example in the minbpe README is another check.

**E7. Tokenizer tests (about 20 minutes)**

1. Round trip: decoding the encoding gives back the original for the empty string, ASCII, accented text, emoji, mixed scripts, combining characters, a very long string and randomly generated Unicode.
2. The encoded length never exceeds the byte length of the text.
3. The final vocabulary size equals the requested size.
4. Training twice on the same text gives identical merges.
5. Report the compression ratio (bytes per token) on train text and on text the tokenizer never saw.

**Done when:** every property above passes and your hand-computed merges match the program.

### Hour 10 · Build · E8–E9: regex splitting, special tokens, persistence, behavior lab

- **Focus:** make the tokenizer behave like a production one, then study how real tokenizers fail.
- **Materials:** the regex and special-token chapters of the tokenizer video; steps 3 and 4 of exercise.md; the GPT-2 paper Section 2.2; [Tiktokenizer](https://tiktokenizer.vercel.app); [tiktoken](https://github.com/openai/tiktoken).

**E8. Pre-tokenization, special tokens, persistence (about 30 minutes)**

1. Split text into chunks by character category (letters, numbers, punctuation, whitespace) with a regex before merging, so merges never cross chunk boundaries. Take the pattern from the GPT-2 and GPT-4 sources instead of inventing one, and note in one sentence what differs between them.
2. Add special tokens, at least an end-of-text token, that live outside the merged vocabulary. Encoding must treat special strings found in ordinary user text as plain text unless the caller explicitly allows them, and a test must prove it.
3. Save the tokenizer (merges, vocabulary, special tokens, split pattern) to disk, reload it, and verify that encodings are identical before and after.

**E9. Tokenizer behavior lab (about 30 minutes)**

1. Train on Tiny Shakespeare at four vocabulary sizes of your choice (spread widely, for example from a few hundred up to a few thousand) and plot compression ratio against vocabulary size.
2. For one trained tokenizer, list the 20 longest tokens and the 20 shortest merged tokens and note anything surprising.
3. Using Tiktokenizer, compare an older and a newer GPT encoding on: leading spaces, a many-digit number, indented code, a non-English sentence and a URL. Write five observations, each tied to a failure an LLM can show because of it (spelling, arithmetic, whitespace sensitivity, weaker non-English performance, wasted context on code).
4. Compare the token counts of your tokenizer and tiktoken on the same English paragraph and explain the gap.

**Done when:** save-and-reload equality passes, the special-token safety test passes, and the lab notes include the plot and five written observations.

### Hour 11 · Build · E10: character-level vs BPE, compared fairly

- **Focus:** wire your tokenizer into your GPT and learn why raw loss values from different tokenizers cannot be compared.
- **Materials:** your E3 model, E8 tokenizer, Tiny Shakespeare; your Hour 4 character-level result as the baseline.

**E10. Controlled comparison**

1. Train the same architecture on Tiny Shakespeare twice: once character-level, once with your BPE tokenizer at a vocabulary size you choose. Hold constant the layer, head and width settings, the optimizer steps and the tokens per batch; write down that the BPE model sees more characters per step.
2. Raw validation loss is per token, so it is not comparable across tokenizers. Convert both to a tokenizer-independent number: total validation negative log-likelihood in bits divided by the number of characters (or bytes) in the validation text.
3. Fill in a results table with: vocabulary size, embedding-table parameters, total parameters, tokens per training epoch, validation loss, bits per character, tokens per second.
4. Write one paragraph on the trade-off you observe (larger vocabulary means shorter sequences and more effective context, but a bigger embedding table and, on a small corpus, possible overfitting).

**Done when:** the table is complete, both models use the same validation text, and your paragraph is backed by your own numbers.

### Hour 12 · Build · E11: your own corpus

- **Focus:** show the pipeline works outside the tutorial dataset.
- **Materials:** [Project Gutenberg](https://www.gutenberg.org/) (public-domain books by one author), [Spider](https://yale-lily.github.io/spider) (concatenate its training SQL and schema text; this feeds the running domain), or your own code or documents if you have the right to use them.

**E11. Train on a corpus you chose**

1. Choose a corpus of roughly 1 MB to 50 MB. Record its source and license in the README.
2. Clean it minimally and document each cleaning rule.
3. Split into train and validation by document or file, not by random lines, so near-duplicate text cannot leak across the split.
4. Train the tokenizer on the training split only, then train the GPT with the same pipeline as E10.
5. Compare validation loss against a trivial baseline (a unigram or bigram model on the same data) to show that attention is adding something.
6. Generate ten samples at three temperatures, and write a short error analysis: repeated phrases, invented words, structure it learned, structure it missed.
7. Record training time, hardware and approximate cost.

**Done when:** a trained checkpoint, the tokenizer files, the baseline comparison and the error analysis are all saved and reproducible from one command.

### Hour 13 · Hardening · E12: tests, types, CI

- **Focus:** turn the exercises into a maintainable package.
- **Materials:** your Week 1 `ml-template`; [pytest](https://docs.pytest.org/), [Hypothesis](https://hypothesis.readthedocs.io/), [ruff](https://docs.astral.sh/ruff/).

**E12. Test suite and CI**

1. Move model, tokenizer and training code into a package, with notebooks importing from it rather than holding logic.
2. Automate the E2, E4 and E7 checks as pytest tests. Express the tokenizer round-trip as a property-based test with Hypothesis.
3. Add a seeded determinism test: the same seed gives the same loss at the first step.
4. Add an overfit-one-batch test on a tiny CPU model that must reach a loss threshold within a fixed number of steps.
5. Mark any test needing a GPU as skipped by default.
6. Make ruff and type checking pass, and have the CI workflow run the suite on CPU in under five minutes.

**Done when:** CI is green on a pull request, and a deliberately broken causal mask makes the leakage test fail.

### Hour 14 · Hardening · E13: tracking, container, README, ADR

- **Focus:** make the work reproducible and presentable.
- **Materials:** [W&B docs](https://docs.wandb.ai/) (runs, Artifacts, Tables); your Week 1 Dockerfile.

**E13. Reproducibility package**

1. Log to W&B: training and validation loss, learning rate, tokens per second, the config, the git SHA. Log generated samples as a table. Store the tokenizer files and best checkpoint as Artifacts.
2. Make the Docker image build and run both the test suite and a 50-step CPU smoke training run.
3. Write the README: what this is, why it exists, how to run each step, the E10 results table, sample outputs, known limitations.
4. Write a one-page ADR on one decision, for example pre-LN vs post-LN, or the BPE vocabulary size you chose, with the evidence you gathered.

**Done when:** a fresh clone reproduces the smoke run using only the README.

### Hour 15 · Review · checklist, notes, self-check

- **Focus:** consolidate, then close the week.
- **Materials:** [Main plan](https://claude.ai/artifact/FD56r2oaE9BXX8VZPiQAeo) Section 7 (universal checklist and the Week 3 row); the [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) as optional reading.
- **Do:** run the checklist in Section 6 below, write five-line notes for CS2 and CS3 in your own words, answer the self-check questions in Section 8 without notes, add a README paragraph on what surprised you, and tag `week-03`. Optionally compare your model file against nanoGPT’s and list five differences.
- **E14. Rebuild challenge (schedule it 2–5 days later, outside the 15 hours’ core work if needed):** from a blank file and with a 45-minute timer, rebuild causal multi-head attention and its tests; then with a 30-minute timer rebuild BPE train, encode and decode. Record what you forgot. If either fails, redo that hour’s code-along section before Week 4.
- **Output:** tag `week-03`, notes, answered self-check, E14 scheduled in your calendar.

## 5. Stretch exercises (only if you are ahead)

1. **Ablation table:** train with each of these removed in turn and record validation loss and training stability: residual connections, LayerNorm, the 1/√d scaling, multi-head (use one head), positional embeddings.
2. **Positional comparison:** learned vs fixed sinusoidal position embeddings on the same run.
3. **Attention maps:** plot attention weights for a few heads on a sample sentence and describe any pattern you see.
4. **Tokenizer parity:** build the GPT-4-style regex tokenizer and compare its output with tiktoken on 20 varied strings; list any mismatches.
5. **SentencePiece:** train a small BPE model with [SentencePiece](https://github.com/google/sentencepiece) and compare its vocabulary with yours.
6. **Permutation experiment:** show that without the mask and positional information, shuffling input tokens shuffles outputs the same way, and that adding the mask breaks that property.

## 6. Definition of done and Week 3 checklist

Week 3 is done when all of these are true:

- E1–E13 complete; E14 scheduled.
- Your attention matches PyTorch’s built-in attention within tolerance, and a test proves the causal mask leaks no future tokens.
- The tokenizer round-trips arbitrary Unicode, handles special tokens safely and reloads identically from disk.
- The E10 table compares character-level and BPE models using bits per character.
- CI is green, the Docker smoke run works from a fresh clone, W&B holds the runs and Artifacts, and the README and ADR exist.
- Repo hygiene: typed code, no logic stranded in notebooks, no hard-coded paths, seeds set, lockfile pinned, no secrets committed.

## 7. Suggested calendar and triage

| Session | Hours | Notes |
| --- | --- | --- |
| A (3 h) | 1–3 | Cheat sheets, bigram, attention |
| B (3 h) | 4–6 | Block and training run, both tokenizer hours |
| C (3 h) | 7–9 | Sleep between B and C. Attention, GPT, BPE v1 |
| D (3 h) | 10–12 | Tokenizer v2 and lab, fair comparison, your corpus |
| E (3 h) | 13–15 | Tests, CI, tracking, Docker, review |

**If you fall behind:** drop Mutate #1, keep only items 2 and 3 of E9, use a Tiny Shakespeare-sized corpus in E11, and skip the optional readings. Never drop E1–E2, E6–E7 or E12; they are the parts that prove understanding and engineering quality.

## 8. Self-check questions (answer without notes)

1. Why does attention cost grow with the square of the sequence length, and what does that imply for the context window?
2. What happens to the softmax at initialization if you skip the scaling by head size, and why does that hurt training?
3. Which dimensions does attention mix information across, and which does it never mix?
4. What information does the causal mask remove, and why is that required for next-token training?
5. Why do deep Transformer stacks need residual connections and normalization, and what changes between pre-LN and post-LN?
6. With the model width fixed, how does changing the number of heads change the parameter count?
7. What is weight tying, and what does it save?
8. Why can byte-level BPE never produce an unknown token?
9. At encode time, why are merges applied in learned order instead of by frequency in the new text?
10. Why can’t you compare validation loss between a character-level model and a BPE model directly, and how do you compare them?
11. Name three LLM quirks caused by tokenization and explain the mechanism behind each.
12. Why does a special token string typed by a user need explicit handling?
