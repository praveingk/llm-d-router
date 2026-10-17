package programaware

// Cold-write budget for the group-turn strategy — a port of the scratch sim's
// WriteBudgetPolicy (~/scratch/tenant-arms/frontier/tenant_arms.py) to the
// router's event model, per task/turn-group-scheduler (p7 verification, p7b
// adversarial review, p9 index calibration).
//
// Mechanism: per-tenant token buckets denominated in COLD-WRITE tokens — the
// KV-cache tokens a dispatch materializes (prompt − predicted-cached). Warm
// dispatches charge ~nothing and are never gated (the ratchet: the budget
// binds only while cold traffic runs, then goes slack in the warm well). An
// unaffordable cold request is deferred PER-REQUEST — never tenant-wide; the
// p7b finding is that tenant-coarse masking amplifies winner-take-most by
// stalling the losing tenant's warm dispatches too — and re-enters ranking on
// the next Pick once its bucket refills. Buckets use DEBT semantics for
// oversized charges (dispatch when the bucket is full and let credit go
// negative, refilled at the configured rate): a cap-only bucket livelocks on
// charges larger than its burst capacity, and the long-run write rate stays
// exactly the configured rate (the scratch documented the livelock and the
// fix; both are preserved here).
//
// DESIGN BLOCKER — the charge source is intentionally UNWIRED. The charge
// basis is the prefix producer's per-request prediction (prompt tokens −
// predicted cached tokens), which p9 calibrated on the fixed index
// (lruCapacityPerServer 10038 + real-token tokenizer) at +0.104 material-
// charge optimism — inside the ≤20% survival bar. But that prediction is
// computed AFTER the flow-control dequeue: the producer runs in
// runDataProducerPlugins (director.go), which executes once Pick has released
// the request, and request.Body.TokenizedRequest is likewise populated only
// by the post-dequeue token producer. Nothing prefix-related exists for a
// request that is still queued, so the prediction is NOT accessible at Pick
// time, where the defer-or-dispatch decision must happen. Config validation
// therefore REFUSES groupTurnColdBudgetTokensPerSec > 0 until a Pick-time
// charge source is wired; coldCharge below is the seam. The bucket, deferral
// and selection mechanics are complete and unit-tested so the wiring is the
// only remaining piece (p7b live-arm spec item 1 is the requirement).
//
// Known deviation from the scratch, by construction of the router's event
// model: the sim scheduled a wake event at the earliest deferral expiry so a
// deferred request re-enters ranking the moment its bucket refills; the
// flow-control fairness API has no timer, so re-entry happens on the next
// Pick — i.e. on the next queue event (arrival, completion). Under load Picks
// are continuous; in a fully quiet system a deferred request waits for the
// next external event. Also, deferral keys the program's FIFO head: later
// requests of the same program cannot jump a deferred head (queue semantics;
// agentic sessions are sequential anyway — one request per program at a
// time).

import (
	"hash/fnv"
	"math"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// coldBudgetWindowSeconds is the bucket burst window: capacity =
	// per-tenant rate × window (the scratch sim's burst_s = 10).
	coldBudgetWindowSeconds = 10.0

	// coldDeferFloor is the minimum deferral duration (the scratch's
	// defer_floor = 0.05 s), so a rounding-sized shortfall cannot cause
	// immediate re-pick churn.
	coldDeferFloor = 50 * time.Millisecond
)

var (
	groupTurnColdCredit = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_cold_budget_credit",
			Help:      metricsutil.HelpMsgWithStability("Cold-write budget bucket credit per tenant (negative = debt).", compbasemetrics.ALPHA),
		},
		[]string{"tenant"})

	groupTurnColdChargedTokens = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_cold_budget_charged_tokens",
			Help:      metricsutil.HelpMsgWithStability("Predicted cold tokens charged to a tenant bucket at a dispatch decision.", compbasemetrics.ALPHA),
		},
		[]string{"tenant"})

	groupTurnColdDefers = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_cold_budget_defers_total",
			Help:      metricsutil.HelpMsgWithStability("Cold requests deferred until their tenant bucket refills (per-request, never tenant-wide).", compbasemetrics.ALPHA),
		},
		[]string{"tenant"})

	groupTurnColdIdlePicks = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_cold_budget_idle_picks_total",
			Help:      metricsutil.HelpMsgWithStability("Picks that dispatched nothing because every waiting request was deferred (honest binding-budget idle).", compbasemetrics.ALPHA),
		})

	groupTurnColdNoPrediction = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_cold_budget_no_prediction_total",
			Help:      metricsutil.HelpMsgWithStability("Charge decisions made without a prediction (fail-open: treated as warm, never gated).", compbasemetrics.ALPHA),
		})
)

// coldChargeEstimator predicts the cold-write tokens (prompt − predicted
// cached) for a request about to be dispatched. ok=false means "no prediction
// available": the request is treated as warm and counted — missing data must
// not gate dispatch (the prompt-denominated failure mode p7b measured
// converts rationing into idleness). The production source is unwired (see
// the file comment); tests inject stubs.
type coldChargeEstimator func(request *fwksched.InferenceRequest) (coldTokens float64, ok bool)

// coldBucket is one tenant's token bucket. It starts full (the scratch
// initializes [rate×burst_s, 0]); credit may go negative (debt semantics).
type coldBucket struct {
	credit     float64
	lastRefill time.Time
}

// coldRateLocked returns the per-tenant refill rate (total budget split
// evenly across tenants).
func (s *groupTurnStrategy) coldRateLocked() float64 {
	return s.coldBudgetTokensPerSec / float64(s.coldBudgetTenants)
}

// coldCapacityLocked returns the bucket burst size: rate × window.
func (s *groupTurnStrategy) coldCapacityLocked() float64 {
	return s.coldRateLocked() * coldBudgetWindowSeconds
}

// coldBucketLocked returns the tenant's bucket, creating it full on first
// touch (callers hold groupMu).
func (s *groupTurnStrategy) coldBucketLocked(tenant int, now time.Time) *coldBucket {
	if s.coldBuckets == nil {
		s.coldBuckets = make(map[int]*coldBucket)
	}
	b, ok := s.coldBuckets[tenant]
	if !ok {
		b = &coldBucket{credit: s.coldCapacityLocked(), lastRefill: now}
		s.coldBuckets[tenant] = b
	}
	return b
}

// coldRefillLocked advances every bucket to now: credit += rate·Δt, capped
// at capacity. Called once per Pick, before the selection loop (the scratch
// refills at the top of pick; within one Pick the credit is frozen except
// for charges).
func (s *groupTurnStrategy) coldRefillLocked(now time.Time) {
	rate := s.coldRateLocked()
	capacity := s.coldCapacityLocked()
	for _, b := range s.coldBuckets {
		if elapsed := now.Sub(b.lastRefill).Seconds(); elapsed > 0 {
			b.credit = math.Min(capacity, b.credit+rate*elapsed)
		}
		b.lastRefill = now
	}
}

// coldThreshold is the debt-semantics affordability threshold: a charge
// larger than the whole bucket is affordable only when the bucket is FULL
// (credit == capacity ≥ threshold), after which credit goes negative and
// refills at rate — the long-run rate stays exact and no oversized request
// is permanently unaffordable.
func coldThreshold(charge, capacity float64) float64 {
	return math.Min(charge, capacity)
}

// coldCanAffordLocked reports whether the tenant's bucket could absorb
// charge right now, WITHOUT mutating it. Used to pre-screen entrants before
// rotating a member out for them: a pre-screened entrant that is never
// dispatched must cost nothing.
func (s *groupTurnStrategy) coldCanAffordLocked(tenant int, charge float64, now time.Time) bool {
	b := s.coldBucketLocked(tenant, now)
	return b.credit >= coldThreshold(charge, s.coldCapacityLocked())
}

// coldTryChargeLocked levies charge on the tenant's bucket at the dispatch
// decision (credit may go negative). Returns false when unaffordable, with
// the bucket untouched.
func (s *groupTurnStrategy) coldTryChargeLocked(tenant int, charge float64, now time.Time) bool {
	b := s.coldBucketLocked(tenant, now)
	if b.credit < coldThreshold(charge, s.coldCapacityLocked()) {
		return false
	}
	b.credit -= charge
	groupTurnColdChargedTokens.WithLabelValues(strconv.Itoa(tenant)).Add(charge)
	return true
}

// coldDeferLocked defers the program's head request until its tenant's
// bucket could afford the charge: need = (threshold − credit) / rate,
// floored at coldDeferFloor. This is the scratch's defer, keyed per request
// (the program's FIFO head) instead of per tenant.
func (s *groupTurnStrategy) coldDeferLocked(id string, charge float64, now time.Time) {
	tenant := s.coldTenantOf(id)
	b := s.coldBucketLocked(tenant, now)
	rate := s.coldRateLocked()
	need := time.Duration(1 << 62) // effectively forever; matches the scratch's 1e9 s
	if rate > 0 {
		need = time.Duration((coldThreshold(charge, s.coldCapacityLocked()) - b.credit) / rate * float64(time.Second))
	}
	if need < coldDeferFloor {
		need = coldDeferFloor
	}
	if s.coldDeferUntil == nil {
		s.coldDeferUntil = make(map[string]time.Time)
	}
	s.coldDeferUntil[id] = now.Add(need)
	groupTurnColdDefers.WithLabelValues(strconv.Itoa(tenant)).Inc()
}

// coldSelectableLocked filters waiting down to the flows whose deferral has
// expired (or that were never deferred), pruning expired entries as it goes:
// the deferred request re-enters ranking on the next Pick after refill.
func (s *groupTurnStrategy) coldSelectableLocked(waiting []groupTurnWaiting, now time.Time) []groupTurnWaiting {
	if s.coldDeferUntil == nil || len(s.coldDeferUntil) == 0 {
		return waiting
	}
	selectable := make([]groupTurnWaiting, 0, len(waiting))
	for _, w := range waiting {
		until, deferred := s.coldDeferUntil[w.id]
		if deferred {
			if until.After(now) {
				continue // still deferred
			}
			delete(s.coldDeferUntil, w.id) // expired: back into ranking
		}
		selectable = append(selectable, w)
	}
	return selectable
}

// coldTenantOf maps a program (fairness ID) to a budget tenant: a stable
// FNV-1a hash modulo the tenant count. The scratch sim split sessions by slot
// parity; the live port reproduces that structure with a stable hash of the
// fairness ID — restart-stable, arrival-order-independent, uniform over the
// session space.
func (s *groupTurnStrategy) coldTenantOf(id string) int {
	h := fnv.New64a()
	_, _ = h.Write([]byte(id))
	return int(h.Sum64() % uint64(s.coldBudgetTenants))
}

// coldChargeForHead evaluates the charge estimator on the head request of a
// waiting program. A missing head/request, or an estimator reporting "no
// prediction", fails open (warm: never gate dispatch on missing data) and is
// counted.
func (s *groupTurnStrategy) coldChargeForHead(waiting []groupTurnWaiting, id string) (float64, bool) {
	noPrediction := func() (float64, bool) {
		groupTurnColdNoPrediction.Inc()
		return 0, false
	}
	if s.coldCharge == nil {
		return noPrediction()
	}
	for _, w := range waiting {
		if w.id != id {
			continue
		}
		head := w.qi.Queue.Peek()
		if head == nil {
			return noPrediction()
		}
		fcr := head.OriginalRequest()
		if fcr == nil {
			return noPrediction()
		}
		request := fcr.InferenceRequest()
		if request == nil {
			return noPrediction()
		}
		cold, ok := s.coldCharge(request)
		if !ok {
			return noPrediction()
		}
		return math.Max(cold, 0), true
	}
	return noPrediction()
}

// coldBudgetOff reports whether the layer is disabled.
func (s *groupTurnStrategy) coldBudgetOff() bool {
	return s.coldBudgetTokensPerSec <= 0 || s.coldBudgetTenants <= 0
}

// pickAffordableEntrantLocked returns the next entrant by entry policy whose
// head the cold budget could afford to dispatch, deferring unaffordable
// candidates (per-request) until their bucket refills. The affordability test
// does NOT charge — the charge is levied only at the actual dispatch
// decision in pickWithColdBudget, so a pre-screened entrant that is never
// dispatched costs nothing. With the budget off it is exactly
// pickEntrantLocked.
func (s *groupTurnStrategy) pickAffordableEntrantLocked(selectable []groupTurnWaiting, now time.Time) string {
	if s.coldBudgetOff() {
		return s.pickEntrantLocked(selectable, now)
	}
	for len(selectable) > 0 {
		ent := s.pickEntrantLocked(selectable, now)
		if ent == "" {
			return ""
		}
		charge, ok := s.coldChargeForHead(selectable, ent)
		if !ok || charge <= 0 || s.coldCanAffordLocked(s.coldTenantOf(ent), charge, now) {
			return ent
		}
		s.coldDeferLocked(ent, charge, now)
		selectable = filterOutWaiting(selectable, ent)
	}
	return ""
}

// filterOutWaiting returns waiting without id, in a fresh slice (the input is
// shared with the caller's selection view).
func filterOutWaiting(waiting []groupTurnWaiting, id string) []groupTurnWaiting {
	out := make([]groupTurnWaiting, 0, len(waiting))
	for _, w := range waiting {
		if w.id != id {
			out = append(out, w)
		}
	}
	return out
}

// coldExportCreditLocked publishes a tenant's bucket credit.
func (s *groupTurnStrategy) coldExportCreditLocked(tenant int, now time.Time) {
	b := s.coldBucketLocked(tenant, now)
	groupTurnColdCredit.WithLabelValues(strconv.Itoa(tenant)).Set(b.credit)
}

// pickWithColdBudget runs the group-turn selection under the cold-write
// budget. Budget off: one plain selection pass over all waiting flows —
// behaviorally identical to the pre-budget strategy. Budget on: propose →
// charge-check → defer, re-selecting without deferred candidates; every
// iteration either dispatches or defers a program that was selectable at loop
// entry, so the loop is bounded by the number of waiting flows. When every
// waiting request is deferred, Pick returns nil — the honest binding-budget
// idle of the scratch sim (the engine waits for a bucket to refill).
func (s *groupTurnStrategy) pickWithColdBudget(waiting []groupTurnWaiting, now time.Time) flowcontrol.FlowQueueAccessor {
	if s.coldBudgetOff() {
		candID := s.selectCandidateLocked(waiting, waiting, now)
		s.setMembershipGauges(waiting)
		if candID == "" {
			return nil
		}
		return waitingQueue(waiting, candID)
	}

	s.coldRefillLocked(now)
	for tenant := range s.coldBuckets {
		s.coldExportCreditLocked(tenant, now)
	}
	for {
		selectable := s.coldSelectableLocked(waiting, now)
		if len(selectable) == 0 {
			// Every waiting request is deferred: idle this cycle. The next
			// Pick — any queue event — re-evaluates once buckets refill (see
			// the file comment for the no-timer-wake deviation).
			groupTurnColdIdlePicks.Inc()
			s.setMembershipGauges(waiting)
			return nil
		}
		candID := s.selectCandidateLocked(waiting, selectable, now)
		if candID == "" {
			groupTurnColdIdlePicks.Inc()
			s.setMembershipGauges(waiting)
			return nil
		}
		charge, ok := s.coldChargeForHead(waiting, candID)
		if !ok || charge <= 0 {
			// Warm (or unpredicted) head: the ratchet — the budget never
			// gates warm dispatches.
			s.setMembershipGauges(waiting)
			return waitingQueue(waiting, candID)
		}
		tenant := s.coldTenantOf(candID)
		if s.coldTryChargeLocked(tenant, charge, now) {
			s.coldExportCreditLocked(tenant, now)
			s.setMembershipGauges(waiting)
			return waitingQueue(waiting, candID)
		}
		s.coldDeferLocked(candID, charge, now)
		// Loop: re-select without the just-deferred candidate.
	}
}
