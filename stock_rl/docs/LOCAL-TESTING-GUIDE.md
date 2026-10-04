# Local Testing Framework - Test Everything Before Cloud

**Goal:** Run 100% of NIFTY-RL stack on a laptop (16GB RAM, no GPU) using Docker Compose + kind (K8s in Docker). Zero cloud cost, zero SEBI risk.

## 1. One-Command Local Stack

```bash
# Prereqs: Docker Desktop, kind, kubectl, helm, python 3.11
make local-up
# Does:
# 1. kind create cluster --config kind-config.yaml
# 2. docker-compose -f docker-compose.local.yaml up -d kafka redis timescaledb neo4j qdrant minio localstack wiremock
# 3. helm install keda kedacore/keda --namespace keda
# 4. tilt up (hot-reload microservices)
```

**docker-compose.local.yaml**
```yaml
services:
  kafka: { image: bitnami/kafka:3.6, ports: ["9092:9092"] }
  redis: { image: redis:7-alpine }
  timescaledb: { image: timescale/timescaledb:2.14-pg15 }
  neo4j: { image: neo4j:5-community, environment: [NEO4J_AUTH=neo4j/test123] }
  qdrant: { image: qdrant/qdrant:v1.8 }
  minio: { image: minio/minio, command: server /data } # S3 mock
  localstack: { image: localstack/localstack } # AWS mock
  zerodha-mock: { image: wiremock/wiremock, volumes: ["./mocks/zerodha:/home/wiremock/mappings"] }
```

## 2. Testing Pyramid

| Level | Tool | Tests | Command |
|---|---|---|---|
| Unit | pytest, jest | Single function | pytest services/ingestion/tests/test_parser.py |
| Contract | Pact | API schema | pact-verifier --provider=feature-store |
| Integration | Testcontainers | Service+DB | pytest -k integration --with-containers |
| Component | Compose | One service + deps | docker-compose run ingestion pytest |
| E2E | Playwright+Kafka | Tick->Signal->Order | pytest tests/e2e/test_tick_to_signal.py |
| Perf | Locust/k6 | Latency p99 <100ms | locust -f tests/perf/locustfile.py |
| Chaos | Chaos Mesh | Kill agent, lag | chaos run kill-sentiment-agent.yaml |
| Compliance | OPA Conftest | robots.txt, audit | conftest test --policy legal/ |

## 3. Per-Service Local Tests

### A. Data Ingestion Service
- Mock: tests/mocks/nse_ticks.json
- Unit: parser handles bonus/split: TCS bonus 1:1 -> 3500 to 1750
- Integration: Testcontainers Kafka - publish tick, assert consumer gets it in 5s
- Legal: test_respects_robots_txt() - must return False for disallowed path, test rate limiter (2 req/sec -> backoff)
- Coverage: --cov-fail-under=80

### B. Sentiment Engine
- Use ollama llama3.1:8b or FakeLLM for local
- Unit: sentiment("Q2 profit beat") >0.5
- E2E: publish to news.raw -> Qdrant has embedding in 10s
- Hallucination: feed "RELIANCE buys Google for $1" -> checker flags credibility <0.3

### C. Knowledge Graph
- Seed: CREATE (ONGC)-[:DEPENDS_ON]->(Crude)
- Unit: get_dependencies("ONGC") == ["Crude"]
- Perf: MATCH (s)-[*1..3]->(e) WHERE e.severity>0.8 <50ms

### D. Multi-Agent Orchestrator
```python
def test_geopolitical_triggers_on_entropy():
    orchestrator = Orchestrator(llm=FakeLLM(["Crude up Iran"]))
    result = orchestrator.run(stock="ONGC", entropy=0.9)
    assert "GDELT" in result.tools_called
    assert result.context["geopolitical_risk"] >0.7
```
- Debate test: Bull vs Bear 3 rounds, judge picks winner

### E. Feature Store
- Unit: zscore([1,2,3]) normalized
- Integration: Feast materialize TimescaleDB->Redis <10ms read
- Data Quality: Great Expectations - RSI 0-100, no future timestamps

### F. RL Training (cheap local)
- Use 1 month, 2 stocks, 100 episodes, CPU only
- Reward includes cost: pnl=100 cost=11 drawdown=5 -> <100
- Deterministic: set_seed(42) twice -> same weights
- World Model MSE <0.01

### G. RL Inference
- ONNX parity: PyTorch vs ONNX diff <1e-5
- Locust: 50 users, p99 <100ms

### H. Risk & Compliance (100% coverage)
```python
def test_circuit_filter(): assert risk.check("RELIANCE", price=2990, circuit=3000) == "REJECT"
def test_fno_ban(): redis.set("fno_ban","ADANIENT"); assert risk.check("ADANIENT") == "REJECT"
def test_kill_switch(): redis.set("kill_switch",1); assert risk.check("ANY") == "REJECT"
def test_audit_immutable(): log_order("123"); s3.delete should raise error (Object Lock)
```

### I. Execution (never real broker)
- WireMock for Zerodha: POST /orders/regular -> 200 mock_123
- Unit: iceberg slicing 1000 -> 10x100
- Integration: signals.validated -> WireMock got call in 5s

### J. Frontend
- Jest + React Testing Library, Playwright
- WS test: mock signal -> table updates <1s
- Kill-switch button -> POST /risk/kill -> red banner

## 4. Spin-Up Logic Testing (KEDA 0->1)

```bash
helm install keda kedacore/keda --namespace keda
kubectl apply -f k8s/keda/sentiment-engine-scaledobject.yaml # minReplicaCount:0
kafka-producer --topic news.raw --messages 15
kubectl get hpa -w # should 0->1 in 30s
# Cron test: 09:00 IST Asia/Kolkata should scale to 1
```

KEDA ScaledObject example:
```yaml
minReplicaCount: 0
maxReplicaCount: 5
triggers:
- type: kafka
  metadata: { topic: news.raw, lagThreshold: "10", bootstrapServers: kafka:9092 }
- type: cron
  metadata: { timezone: Asia/Kolkata, start: 0 9 * * 1-5, end: 30 15 * * 1-5, desiredReplicas: "1" }
```

## 5. Full E2E: Tick -> Mock Order

```bash
make e2e-local
# 1. docker-compose.local up -d
# 2. seed TimescaleDB 1 day Nifty + Neo4j graph
# 3. publish 100 mock ticks + 2 news "Crude up 5% Iran"
# 4. wait: Feature -> Orchestrator -> Inference -> Risk -> Execution
# 5. assert: WireMock order, Qdrant 2 embeddings, MinIO audit 1 entry, Frontend WS signal
# 6. chaos: kill sentiment-engine pod, assert degraded price-only signal still works
```

Expected:
```
E2E PASS: 100 ticks -> 12 signals -> 8 passed risk -> 8 mock orders
Latency p99: 87ms PASS
Audit immutable: PASS
Legal robots.txt: PASS
Cost: $0
```

## 6. CI Before Cloud

.github/workflows/local-test.yaml:
- kind create cluster
- make local-up && make e2e-local && make perf-test && make compliance-test
- trivy image scan
- Only if passes, ArgoCD deploys to EKS prod

## 7. Makefile

```makefile
local-up: kind create cluster; docker-compose -f docker-compose.local.yaml up -d; tilt up
local-down: kind delete cluster; docker-compose down -v
test-unit: pytest services/*/tests/test_*.py --cov
test-integration: pytest -k integration --with-containers
e2e-local: pytest tests/e2e/test_tick_to_signal.py -s
perf-test: locust -f tests/perf/locustfile.py --headless --run-time 2m --check p99<100
compliance-test: conftest test --policy policies/legal/; python tests/legal/test_robots.py
chaos-test: kubectl apply -f chaos/kill-agent.yaml; pytest tests/chaos/test_degraded_mode.py
```

Golden Rule: If it doesn't pass `make e2e-local` on laptop, it never goes to cloud. Saves ~15k INR/month GPU and prevents SEBI issues.
