# Weighted LAS Fairness Policy

**Type:** `weighted-las-fairness-policy`

The Weighted LAS policy serves the flow with the least attained service **per unit of share weight**. Each flow accrues service as its requests complete; dividing that service by a weight declared per request drives the flows' service shares toward the ratio of their weights. At equal weights it is plain least-attained-service, which shares the pool evenly by work done rather than by turn.

Registered at **Alpha** stability, so the EPP requires `--allow-experimental-plugins`. Without it the EPP refuses to start — a clean refusal, not a silent fallback to another fairness policy.

## Why choose this policy?

Because the resource you are dividing is **work**, not attention. [Round Robin](../roundrobin/README.md) gives every flow an equal *turn*, which is the right unit only when requests cost the same. When tenants choose their own prompt sizes, a turn is worth whatever the tenant decides to make it: measured on an 8×H200 GLM-5.3-Flash pool with two tenants at identical rates and identical weights, where one sent 4.6× larger prompts, round robin let that tenant take **81% of the GPU work** while this policy held it to **59%**.

The unit is also configurable, so "work" can mean prompt processing, generation, request count, or a blend — see [Configuration](#configuration).

## What it does

1.  **Scoring**: For each active flow it computes `attainedService / flowWeight`, range-normalizes across the candidates, inverts it (less service per unit of weight scores higher), and blends in the head item's queue wait as an anti-starvation term.
2.  **Charging**: A flow is charged when one of its requests completes (`ResponseBody`, at end of stream), using the configured cost function over the server-reported prompt and completion token counts.
3.  **Decay**: Attained service decays exponentially, so a flow that goes quiet recovers priority. Decay is applied lazily on read or write, so an idle flow ages out without being visited.
4.  **Weight resolution**: The weight is read from the head request's `x-llm-d-inference-fairness-weight` header, clamped to `maxFlowWeight`, and remembered per flow so a tenant need not stamp every request.
5.  **Pruning**: Flows idle beyond `idleTtlSeconds` are swept out, at most once per `sweepSeconds`, on the `Pick` path — the policy owns no goroutine.

## Unit of Fairness

**Service**, defined by the configured cost function:

```
cost = costPerRequest + costPerInputToken × promptTokens + costPerOutputToken × completionTokens
```

The default `0 / 1 / 2` weights an output token at twice an input token.
However, the cost can be used to effect different notions of fairness, because it selects *which* resource the weights divide. Set `0 / 0 / 1` and cost becomes completion tokens alone, so weights 7:3 divide **output tokens per second** 70:30 and prompt processing is free. Set `1 / 0 / 0` and cost becomes a flat charge per request, so the same weights divide **turns** 70:30 however expensive each turn is. So pick the coefficients from the SLO you are dividing, not by tuning — see [Choosing the cost coefficients](#choosing-the-cost-coefficients).

**A flow's weight is keyed by `FlowKey`, which is composite.** One tenant sending at two priorities is two flows with separate service — bands dispatch in strict order, and pooling them would let work done in one band deprioritize the tenant in another.

## Inputs consumed

*   **Queue state**: Iterates active queues on the `PriorityBandAccessor`, reading each head item and its enqueue time.
*   **Request headers**: `x-llm-d-inference-fairness-weight` off the head request; `x-llm-d-inference-fairness-id` supplies the flow id, falling back to a single default id when absent.
*   **Response usage**: `PromptTokens` and `CompletionTokens` from the final response chunk.

## Configuration

```yaml
- type: weighted-las-fairness-policy
  name: weighted-las-fairness-policy
  parameters:
    costPerInputToken: 0      # share generation throughput instead of total work
    costPerOutputToken: 1
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `weightService` | `0.8` | Weight of the normalized service term in the score. |
| `weightHeadWait` | `0.2` | Weight of the head-of-queue wait term. Anti-starvation; it is **not** scaled by the flow weight. |
| `halfLifeSeconds` | `60` | Exponential decay half-life for attained service. `0` disables decay. |
| `maxFlowWeight` | `10` | Ceiling on a declared weight. The header is client-supplied, so without a cap one tenant could monopolize the band. |
| `idleTtlSeconds` | `3600` | Idle flows are pruned after this. |
| `sweepSeconds` | `300` | Minimum interval between idle sweeps. |
| `costPerRequest` | `0` | Flat cost charged per completed request. |
| `costPerInputToken` | `1` | Cost per prompt token. |
| `costPerOutputToken` | `2` | Cost per completion token. |

All three cost coefficients must be `>= 0` and **not all zero** — all-zero charges nothing, so every flow would sit at zero attained service forever and the policy would degenerate into head-wait-only ordering while still looking configured. That is rejected at startup.

### Choosing the cost coefficients

The coefficients choose *what* is being shared, and the choice matters more than it sounds. On a long-context agentic corpus (ISL ~130k, OSL ~1k) input is **97.7%** of the default cost, so the default effectively shares prefill and generation throughput is almost unrepresented.

| `perRequest` / `perInput` / `perOutput` | Shares |
| --- | --- |
| `0 / 1 / 2` | default — roughly GPU work, output weighted 2× |
| `0 / 0 / 1` | output tokens per second |
| `0 / 1 / 0` | prefill only |
| `1 / 0 / 0` | request counts (turns) |
| `0 / 0.1 / 1` | mostly generation, still charging something for prefill |

Measured with `0 / 0 / 1` at weights 7:3, on two tenants with identical prompt and output lengths so that the weight was the only variable: **66.9%** of output tokens went to the weight-7 tenant against a 70% target, confirmed independently by the attained-service gauge at 66.8%. Expect the coefficients to set the split's *direction and rough magnitude*, with a few points of shortfall — see [Trade-offs](#trade-offs).

**Charging nothing for input makes prefill free**, which re-opens the vulnerability that this policy otherwise closes: a tenant can inflate prompt size at no cost. Use `0 / 0 / 1` when the SLO you are dividing is tokens/sec *and* prompt sizes are uniform or trusted.

## Observability

| Metric | Type | Labels |
| --- | --- | --- |
| `llm_d_epp_weighted_las_attained_service_tokens` | gauge | `flow_id`, `priority`, `policy` |
| `llm_d_epp_weighted_las_flow_weight` | gauge | `flow_id`, `priority`, `policy` |

The weight gauge is the only way to confirm the EPP actually **read** a declared weight rather than falling back to the default — without it, a misdelivered header is indistinguishable from a policy that ignores weights. Collectors are per-instance and registered through `handle.Metrics()`, so two instances can share one registry.

Note the service gauge is the *decayed* value, so it reflects recent service rather than a run total, and comparing it across two different `halfLifeSeconds` settings compares two different instruments.

## Trade-offs

*   **It does not fully enforce its target.** Measured at 7:3 on a contended pool, achieved service ratios landed 13–20% short of the ideal, consistently under-correcting toward parity. Four causes have been ruled out — workload heterogeneity, the head-wait term (with only two flows it cannot flip a decision), starvation, and decay/lag as a *major* factor. The residual is open. Treat the weights as a strong bias, not a hard guarantee.
*   **The control loop is delayed.** Service is charged at completion, which on long requests is tens of seconds after the dispatch decision it should have informed. [Round Robin](../roundrobin/README.md) knows a turn's cost at selection time and has no such lag.
*   **Weights are sticky per flow.** A flow's last valid declared weight stands for requests that omit the header. Convenient, but an A/B comparison **must** restart the EPP (or use fresh flow ids) between arms, or the unweighted arm silently inherits the previous arm's weights and reports a weighted split while appearing to send none.

## Related Documentation
*   [Fairness Overview](../README.md)
*   [Round Robin](../roundrobin/README.md) — the same idea with turns as the unit
*   [Program-Aware](../program-aware/README.md) — its `las` strategy shares this service definition
*   [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
