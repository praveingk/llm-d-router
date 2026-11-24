# Session Observer

**Type:** `session-observer`
**Interfaces:** `requestcontrol.RequestHeaderProcessor`, `requestcontrol.PreRequest`, `requestcontrol.ResponseBodyProcessor`

A **passive** observability plugin that records per-session usage metrics — how many
sessions are active, from which client harness, how many turns they run, how long
they live — without influencing any request decision. It is the metrics-only
sibling of `agent-identity`: where `agent-identity` writes an attribute consumed by
flow control, `session-observer` writes nothing at all.

## What It Does

For every request the plugin resolves two labels and maintains a bounded session
table:

- **`session_header`** — the name of the first header in the `sessionHeaders`
  priority list whose trimmed value is non-empty (`none` when nothing matched,
  `invalid` when the value is longer than `maxSessionIdBytes` or non-printable).
  Requests labelled `none`/`invalid` are counted, but not tracked as sessions.
- **`harness`** — the product token before the first `/` or space from the first
  non-empty `harnessHeaders` header (`x-client-user-agent`, then `user-agent`),
  lowercased, at most 64 characters of `[a-z0-9._-]`; unparsable values and values
  beyond the `maxHarnessLabels` cap collapse to `other`, and requests carrying no
  harness header at all count as `unknown`.

Per session (keyed by a **salted hash**, never the raw ID) it tracks first/last
seen, turn count and in-flight requests. The work is split across three hooks:

- **`RequestHeader`** (before admission) counts requests, bytes, turns and
  session creation, and updates `lastSeen`.
- **`PreRequest`** (after the scheduler picked an endpoint) begins in-flight
  accounting, registering the request ID and applying the 0→1 transition of the
  in-flight gauge. Starting here — not in `RequestHeader` — means requests
  rejected before dispatch (flow-control shed and 429s under saturation,
  admission denial, no endpoints, scheduler error) are counted but never
  inflated into `sessions_active{window="inflight"}`, which would otherwise
  over-report for the 900 s expiry TTL exactly while the system is shedding.
- **`ResponseBody`** with `EndOfStream=true` settles the request. The director
  invokes it for every request that picked a pod — including failed, aborted,
  evicted and disconnected streams (`TerminationCause` carries which).

A dispatched request whose completion somehow never fires (a crash between
hooks, a teardown the EPP never observes) is expired by the sweep after 900 s
and counted under the `expired` eviction reason. The decrement is idempotent.

A background sweep (every `gaugeRefreshSeconds`, stopped with the plugin handle's
context) evicts sessions idle past `sessionIdleTimeoutSeconds` with no in-flight
requests, expires orphaned in-flight entries, and recomputes the gauge series.
Inserting beyond `maxSessions` evicts the least recently active **idle** session;
sessions with in-flight requests are never evicted.

## Passivity Guarantees

- Writes **no** request attribute (does not implement `plugin.ProducerPlugin`).
- Never mutates `request.Headers`, `FairnessID`, or anything else on the request.
- Declares no data dependency, so it cannot reorder or gate other plugins.
- Fails closed to counting-only: no code path can reject or alter a request.

## Privacy

Session IDs are never stored raw and never appear as (or in) a metric label. The
table is keyed by the first 16 bytes of `SHA-256(per-process random salt || id)`.
The salt is generated once per process, so hashes cannot be correlated across
restarts or precomputed by anyone who knows the ID format. Only the derived labels
above (`harness`, `session_header`, `window`, `reason`) ever reach Prometheus.

## Metrics

All metrics use the `llm_d_epp` subsystem and appear on the EPP's `:9090`
`/metrics` endpoint:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_d_epp_session_requests_total` | counter | `harness`, `session_header` | Observed requests. |
| `llm_d_epp_session_request_bytes_total` | counter | `harness`, `session_header` | Observed request body bytes. |
| `llm_d_epp_sessions_active` | gauge | `harness`, `window` | Sessions with at least one request **dispatched to a model server and not yet finished** (`window="inflight"`; requests queued in flow control are not in flight), and sessions whose last request arrived within each configured window (`300`→`5m`, `3600`→`1h`). |
| `llm_d_epp_sessions_started_total` | counter | `harness` | Newly tracked sessions. |
| `llm_d_epp_session_turns` | histogram | `harness` | Turns per evicted session. |
| `llm_d_epp_session_duration_seconds` | histogram | `harness` | Active duration per evicted session. |
| `llm_d_epp_session_table_evictions_total` | counter | `reason` | `idle`, `capacity` or `expired` (see above). |
| `llm_d_epp_session_table_size` | gauge | — | Sessions currently tracked. |

A session's `harness` is fixed at its first request; per-request counters always
use the current request's parsed harness.

## Configuration

**Location:** Top-level `plugins:` list in the `EndpointPickerConfig`.
**Enabled by default:** No. Add a `- type: session-observer` entry to enable; the
runner discovers it as a `RequestHeaderProcessor` and `ResponseBodyProcessor`.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `sessionHeaders` | `[]string` | No | `x-claude-code-session-id`, `x-opencode-session-id`, `x-session-id`, `session-id`, `session_id`, `x-session-affinity` | Priority-ordered session-ID headers; first non-empty match wins, and the matched name becomes the `session_header` label. |
| `harnessHeaders` | `[]string` | No | `x-client-user-agent`, `user-agent` | Priority-ordered harness headers. |
| `maxHarnessLabels` | `int` | No | `50` | Cap on distinct `harness` label values; overflow collapses to `other`. |
| `activeWindowsSeconds` | `[]int` | No | `[300, 3600]` | `sessions_active` window sizes in seconds. |
| `sessionIdleTimeoutSeconds` | `int` | No | `3600` | Idle time after which a session with no in-flight requests is evicted. |
| `maxSessions` | `int` | No | `100000` | Tracked-session cap; the least recently active idle session is evicted for a new one. |
| `gaugeRefreshSeconds` | `int` | No | `15` | Sweep interval: eviction and gauge recompute cadence. |
| `maxSessionIdBytes` | `int` | No | `256` | Session-ID length bound; longer values count as `invalid` and are not tracked. |

Every parameter must be a positive integer where numeric, and the header lists
must be non-empty after defaults are applied.

### Example

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: session-observer
    name: session-observer
    parameters:
      sessionHeaders: [x-claude-code-session-id, x-opencode-session-id, x-session-id, session-id, session_id, x-session-affinity]
      harnessHeaders: [x-client-user-agent, user-agent]
      maxHarnessLabels: 50
      activeWindowsSeconds: [300, 3600]
      sessionIdleTimeoutSeconds: 3600
      maxSessions: 100000
      gaugeRefreshSeconds: 15
```

## Limitations

- **Dispatched-but-unobserved completions leak for up to 900 s.** A request
  whose completion the EPP never observes (crash between hooks, an unobserved
  teardown) keeps its in-flight entry until the sweep expires it (reason
  `expired`). Requests rejected before dispatch never carry an entry at all.
- **Session identity is header-derived.** Clients that strip or rotate session
  headers appear as separate sessions; nothing deduplicates them.
- **The table is per-replica.** Each EPP replica observes only the requests it
  routes; aggregate with `sum()` over replicas.
