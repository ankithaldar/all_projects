# RL-Based Optimization Framework for Crafting Systems

## 1. Frame the Crafting System as an Economic Production Network

Before introducing reinforcement learning (RL), model the crafting system as a **deterministic production network with constraints**.

### 1.1 Core Structural Elements

* **Item set**: All craftable items organized by tier
* **Recipes**: Fixed input–output mappings
* **Resources**: Coins, crafting time, slots, energy
* **Constraints**: Queue limits, parallelism, cooldowns
* **Progression targets**: Items that gate advancement

Model the system as a **directed acyclic graph (DAG)**:

* Nodes = items
* Edges = dependencies
* Weights = time, coin cost, slot usage

Validate:

* No cyclic dependencies
* Consistent scaling across tiers

---

## 2. Establish Benchmarks and Baselines

### 2.1 Deterministic Baselines

Simulate non-learning strategies:

* **Greedy (lowest-tier first)**
* **Critical path prioritization**
* **Coin-minimizing policy**
* **Time-minimizing policy**

Track:

* Time-to-tier progression
* Total coin usage
* Slot utilization
* Craft blocking frequency

These define your **performance envelope**.

### 2.2 Empirical Player Baselines

If data is available:

* Segment player cohorts (fast vs slow progression)
* Analyze time-to-tier distributions
* Measure coin burn patterns

This helps define:

* Target optimization goals
* Acceptable variance

---

## 3. Define Optimality and Inefficiency

### 3.1 Multi-Objective Optimality

Objectives:

* Minimize time-to-progression
* Minimize coin expenditure
* Maintain system stability

Expect **Pareto trade-offs**, not a single optimum.

### 3.2 Inefficiency Metrics

Define inefficiency as avoidable waste:

* Overproduction of unused items
* Idle crafting slots
* Resource starvation
* Misaligned crafting priorities

Key metrics:

* Coins per progression unit
* Slot idle ratio
* Inventory imbalance
* Queue blocking frequency

---

## 4. Detect and Quantify Bottlenecks

### 4.1 Static Graph Analysis

Analyze the DAG:

* Longest (critical) paths
* High fan-in nodes
* Tier depth variance

These reveal **structural bottlenecks**.

### 4.2 Simulation-Based Detection

Run baseline simulations and observe:

* Queue wait times
* Inventory fluctuations
* Craft blocking frequency

Indicators:

* Persistent shortages of specific items
* Repeated delays at the same tier
* Cascading dependency failures

### 4.3 Resource and Queue Dynamics

Measure:

* Slot utilization rate
* Craft completion vs consumption lag
* Coin depletion events

Stress-test by:

* Reducing income rates
* Increasing craft times
* Removing key recipes

This reveals **system fragility**.

---

## 5. RL-Compatible Formulation

### 5.1 State Space

Include:

* Inventory (compressed representation)
* Active crafting queue
* Available slots
* Coin balance
* Dependency deficits

Avoid unnecessary granularity—use sufficient statistics.

### 5.2 Action Space

Define high-level decisions:

* Start crafting recipe X
* Reserve or delay crafting
* Reprioritize queue

Restrict to valid, feasible actions.

### 5.3 Reward Design

Align rewards with inefficiency metrics:

* Penalize idle slots
* Penalize coin expenditure
* Reward critical path progress
* Reward tier completion

Optional shaping:

* Reduction in remaining dependency depth
* Reduction in estimated completion time

### 5.4 Constraints

Enforce:

* Hard constraints (resources, slots)
* Soft constraints (economy pacing)

Mechanisms:

* Action masking
* Penalty functions
* Episode termination rules

---

## 6. Validation Strategy

Before deployment:

* Compare RL policy vs baselines
* Check for exploitative strategies
* Stress-test under parameter shifts

Ensure:

* Bottlenecks are reduced
* System stability improves
* Player experience is smoother

---

## 7. Modeling Assumptions to Revisit

Continuously validate:

* Recipe stability
* Player behavior abstraction accuracy
* Trade-offs between efficiency and monetization

---

## Key Takeaway

RL should be applied **after** the crafting system is well-understood analytically. It functions best as a **policy optimizer over a validated economic model**, not as a discovery mechanism for poorly defined systems.