/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package programaware

import (
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const groupTurnStrategyName = "group-turn"

// Entry policies decide which waiting non-member joins the group next.
const (
	groupTurnEntryLeastService = "least-service"
	groupTurnEntryOldestWait   = "oldest-wait"
)

// Victim policies decide which member gives up its seat when the group is
// fully idle while outsiders wait.
const (
	groupTurnVictimLongestIdle  = "longest-idle"
	groupTurnVictimLeastService = "least-service"
)

// Credit policies decide whether a program leaving the group keeps its
// accrued service (used for re-entry ranking) or restarts from zero.
const (
	groupTurnCreditKeep  = "keep"
	groupTurnCreditReset = "reset"
)

var (
	groupTurnGroupSize = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_group_size",
			Help:      metricsutil.HelpMsgWithStability("Current number of programs in the turn-group.", compbasemetrics.ALPHA),
		})

	groupTurnWaitingOutside = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_waiting_outside",
			Help:      metricsutil.HelpMsgWithStability("Waiting programs that are not group members.", compbasemetrics.ALPHA),
		})

	groupTurnAdmissions = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_admissions_total",
			Help:      metricsutil.HelpMsgWithStability("Programs admitted into the turn-group.", compbasemetrics.ALPHA),
		})

	groupTurnSwaps = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_swaps_total",
			Help:      metricsutil.HelpMsgWithStability("Quota-exhausted members rotated out for a waiting outsider.", compbasemetrics.ALPHA),
		})

	groupTurnVictimEvictions = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_victim_evictions_total",
			Help:      metricsutil.HelpMsgWithStability("Idle members evicted to seat a waiting outsider.", compbasemetrics.ALPHA),
		})

	groupTurnHoldExits = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_hold_exits_total",
			Help:      metricsutil.HelpMsgWithStability("Members that left the group after exceeding the hold time.", compbasemetrics.ALPHA),
		})

	// Label-free by design, matching the other group-turn counters: a
	// per-program label would be user-controlled cardinality (p14 §9.8).
	groupTurnMaxWaitPicks = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_max_wait_picks_total",
			Help:      metricsutil.HelpMsgWithStability("Waiting sessions served via the max-wait bound (D), preempting warmth-EDF.", compbasemetrics.ALPHA),
		})

	groupTurnServiceTokens = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_service_tokens",
			Help:      metricsutil.HelpMsgWithStability("Time-decayed attained service per program (turn-group entry ranking).", compbasemetrics.ALPHA),
		},
		[]string{"program_id"})
)

// groupTurnState is the per-program state kept by the turn-group strategy.
// It deliberately duplicates the in-flight/dispatch accounting (rather than
// reading ProgramMetrics) so membership decisions can reason about programs
// whose queues are not present in the current priority band.
type groupTurnState struct {
	mu sync.Mutex

	// service is LAS-style attained service (prompt + 2*completion tokens),
	// lazily decayed against decayAnchor.
	service     float64
	decayAnchor time.Time

	// meanCost/completions track this program's running mean completion cost,
	// used to estimate the service still pending on in-flight requests.
	meanCost    float64
	completions int64

	dispatchedTotal int64     // all requests ever dispatched for the program
	stayStart       int64     // dispatchedTotal snapshot when the current stay began
	lastDispatch    time.Time // for least-recently-served ordering among members
	lastActivity    time.Time // last dispatch or completion; drives hold time and victim choice
	lastPrompt      float64   // engine-reported prompt tokens of the last completed request (cold-charge scale)
	inFlight        int64
	pendingGap      time.Duration // inter-dispatch gap captured at dispatch (τ estimator input)
}

// decayLocked folds the decay accrued since decayAnchor into service and
// advances the anchor to now. halfLifeSeconds <= 0 disables decay. The
// caller must hold mu.
func (st *groupTurnState) decayLocked(now time.Time, halfLifeSeconds float64) {
	if halfLifeSeconds > 0 && !st.decayAnchor.IsZero() {
		if elapsed := now.Sub(st.decayAnchor).Seconds(); elapsed > 0 {
			st.service *= math.Pow(0.5, elapsed/halfLifeSeconds)
		}
	}
	st.decayAnchor = now
}

func (st *groupTurnState) Service(now time.Time, halfLifeSeconds float64) float64 {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.decayLocked(now, halfLifeSeconds)
	return st.service
}

// AddService charges a completed request's cost. The service accrued so far
// is decayed up to now BEFORE the new cost lands, so a member's service ages
// between its own completions exactly as it does while idle — the same LAS
// semantics the simulation used. Charging without decaying first froze a
// member's service at its stay's peak: two charges of 1000 ten half-lives
// apart read 2000 where LAS reads ~1001, so a rotated-out member re-entered
// the entry contest several half-lives heavier than its true decayed service
// (p14 §3e).
func (st *groupTurnState) AddService(cost float64, now time.Time, halfLifeSeconds float64) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.decayLocked(now, halfLifeSeconds)
	st.service += cost
	st.completions++
	st.meanCost += (cost - st.meanCost) / float64(st.completions)
}

// EffectiveService is the entry-ranking currency: charged service plus an
// estimate of the service still pending on in-flight requests (in-flight
// count times the program's mean completion cost, or the global mean when the
// program has no completion history). Without the pending term, a member that
// just rotated out with un-completed requests would look as light as a
// newcomer and jump straight back in, making the turn quota meaningless. The
// estimate dissolves on its own as real charges land at completion.
func (st *groupTurnState) EffectiveService(now time.Time, halfLifeSeconds, fallbackMean float64) float64 {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.decayLocked(now, halfLifeSeconds)
	mean := fallbackMean
	if st.completions > 0 {
		mean = st.meanCost
	}
	return st.service + float64(st.inFlight)*mean
}

func (st *groupTurnState) ResetService() {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.service = 0
	st.decayAnchor = time.Now()
}

func (st *groupTurnState) RecordDispatch(now time.Time) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.dispatchedTotal++
	st.inFlight++
	st.lastDispatch = now
	st.lastActivity = now
}

// CapturePendingGap records the inter-dispatch gap for the request about to
// be dispatched (zero when there was no prior dispatch — a first request is
// not a re-entry and feeds no estimator bin).
func (st *groupTurnState) CapturePendingGap(now time.Time) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.lastDispatch.IsZero() {
		st.pendingGap = 0
		return
	}
	st.pendingGap = now.Sub(st.lastDispatch)
}

func (st *groupTurnState) takePendingGap() time.Duration {
	st.mu.Lock()
	defer st.mu.Unlock()
	g := st.pendingGap
	st.pendingGap = 0
	return g
}

func (st *groupTurnState) RecordCompletion(now time.Time, promptTokens float64) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.inFlight--
	st.lastActivity = now
	st.lastPrompt = promptTokens
}

// RecordAbort settles a dispatched request whose stream terminated without
// completing (client disconnect, eviction, upstream error): the in-flight
// seat is released and lastActivity advances so hold-time accounting stays
// honest. Nothing else moves — no service is charged (the true cost is
// unknown), the completion mean is not updated, and lastPrompt keeps its
// previous value because an aborted request's prefix was likely never
// cached (p14 §9.1). The inFlight floor guard makes a double-settle from any
// unforeseen framework path a no-op instead of a negative count, which would
// corrupt the pending-service estimate the same way a leak does.
func (st *groupTurnState) RecordAbort(now time.Time) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.inFlight > 0 {
		st.inFlight--
	}
	st.lastActivity = now
}

func (st *groupTurnState) BeginStay(now time.Time) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.stayStart = st.dispatchedTotal
	st.lastActivity = now
}

func (st *groupTurnState) snapshot() (dispatchedTotal, stayStart, inFlight int64, lastDispatch, lastActivity time.Time) {
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.dispatchedTotal, st.stayStart, st.inFlight, st.lastDispatch, st.lastActivity
}

var _ Strategy = &groupTurnStrategy{}

// groupTurnStrategy schedules turns, not requests. A group of programs
// holds dispatch priority — the KV-warm resident set — and membership
// rotates (unbounded by default, bounded for experiments):
//
//   - Pick dispatches the least-recently-served member with a waiting request.
//   - A member that has dispatched its turn quota (turnQuota requests since its
//     stay began) is rotated out in favor of the entry-queue head whenever
//     outsiders wait; the swap happens at pick time so the leaver's prefix has
//     aged one tool gap, aligning the victim with what the engine's LRU would
//     evict anyway.
//   - Entry ranking is least-service (default) or oldest-wait.
//   - When every member is idle while outsiders wait, a victim (longest-idle
//     by default) gives up its seat to the entry-queue head.
//   - A member with nothing pending or in-flight for longer than the hold time
//     leaves on its own, freeing the seat.
//   - Rotated-out and evicted programs keep their (decayed) service by default,
//     so a program away long enough re-enters with near-newcomer priority
//     exactly when its prefix is cold anyway.
//   - Above all of it sits the max-wait bound D: any waiting session whose
//     head request has waited >= D is served next, oldest head first. D is
//     the fairness contract's hard SLO — it preempts warmth-EDF on purpose,
//     because bounding the overload tail is worth more than the cache hit of
//     one more warm refresh (p14 §2: the tail past a few minutes is nearly
//     free to cut).
type groupTurnStrategy struct {
	// groupSize bounds membership for bounded-group experiments. <= 0 (the
	// default) is structurally unbounded: see effectiveGroupSize.
	groupSize       int
	turnQuota       int64
	entryPolicy     string
	victimPolicy    string
	creditPolicy    string
	holdDuration    time.Duration
	halfLifeSeconds float64
	// urgencySeconds enables warmth-EDF member ordering when > 0: warm members
	// (next dispatch still inside the decay knee) are served first, earliest
	// expiry first — dispatching re-reads and refreshes a prefix that is about
	// to age out. Cold members (deadline passed, loss sunk) fall back to LRS.
	// When adaptive is on, the knee is controller-driven and this is the
	// bootstrap/fallback value.
	urgencySeconds time.Duration

	// maxWait is the hard SLO on queue wait (D). Any waiting session — member
	// or not — whose head request has waited >= D is served next, oldest head
	// first, preempting warmth-EDF and the entry contract (an expired
	// outsider is dispatched without membership, like the all-idle gap
	// fill). <= 0 disables the bound.
	maxWait time.Duration

	// Self-balancing layer (strategy_groupturn_sb.go): when adaptive, the
	// group size tracks the engine's running set and the warmth knee tracks
	// the measured re-entry hit curve. Fairness is never adaptive.
	adaptive bool
	gMax     int
	kneeMax  time.Duration
	sb       *sbState

	// Cold-write budget (strategy_groupturn_cold.go): per-tenant token
	// buckets over predicted cold tokens with per-request deferral.
	// coldBudgetTokensPerSec <= 0 (the default) disables the layer entirely;
	// coldCharge is the prediction seam (nil = no source wired — config
	// validation refuses a nonzero budget in that state). Everything below
	// is guarded by groupMu: it is touched only inside Pick's critical
	// section and in EvictProgram.
	coldBudgetTokensPerSec float64
	coldBudgetTenants      int
	coldCharge             coldChargeEstimator
	coldBuckets            map[int]*coldBucket
	coldDeferUntil         map[string]time.Time

	state sync.Map // program ID (string) -> *groupTurnState

	groupMu sync.Mutex
	group   map[string]struct{} // member IDs; guarded by groupMu

	// Global running mean of completion costs; bootstraps the pending-service
	// estimate for programs with no completion history. promptSum/promptCount
	// carry the same statistics for PROMPT tokens — the cold-charge scale for
	// newcomer sessions under the knee-keyed estimator.
	pendingMu    sync.Mutex
	pendingSum   float64
	pendingCount int64
	promptSum    float64
	promptCount  int64
}

// waitingOutsideCount returns how many queued non-member sessions are waiting
// right now (the G controller's drain-freeze input).
func (s *groupTurnStrategy) waitingOutsideCount(queues map[string]QueueInfo) int {
	s.groupMu.Lock()
	defer s.groupMu.Unlock()
	n := 0
	for id, qi := range queues {
		if qi.Len > 0 {
			if _, ok := s.group[id]; !ok {
				n++
			}
		}
	}
	return n
}

// effectiveGroupSize returns the current membership bound (controller-driven
// when adaptive, the configured size otherwise). A configured size <= 0
// means unbounded — treated as infinite at the one place the bound is
// checked (the seat-fill loop), which admits every waiting session and so
// structurally disables the group machinery rather than special-casing
// each site: with no non-members, the quota swap never fires (an exhausted
// member renews its stay), and the all-idle victim path never triggers (a
// member is always waiting) (p15 §3: a large sentinel was rejected because
// it re-binds the machinery with wrong semantics once exceeded).
func (s *groupTurnStrategy) effectiveGroupSize() int {
	if s.adaptive && s.sb != nil {
		return s.sb.effectiveG()
	}
	if s.groupSize <= 0 {
		return math.MaxInt
	}
	return s.groupSize
}

// warmKnee returns the warmth-EDF knee in force right now.
func (s *groupTurnStrategy) warmKnee() time.Duration {
	if s.adaptive && s.sb != nil {
		return s.sb.knee()
	}
	return s.urgencySeconds
}

const groupTurnDefaultMeanCost = 1000

// groupTurnDefaultMeanPrompt bootstraps the cold charge for newcomer sessions
// (no completion history yet) until real completions establish the running
// mean — mid-range for agentic contexts (p0a: mean ~34k, p75 50k).
const groupTurnDefaultMeanPrompt = 32768.0

func (s *groupTurnStrategy) globalMeanCost() float64 {
	s.pendingMu.Lock()
	defer s.pendingMu.Unlock()
	if s.pendingCount == 0 {
		return groupTurnDefaultMeanCost
	}
	return s.pendingSum / float64(s.pendingCount)
}

func (s *groupTurnStrategy) globalMeanPrompt() float64 {
	s.pendingMu.Lock()
	defer s.pendingMu.Unlock()
	if s.promptCount == 0 {
		return groupTurnDefaultMeanPrompt
	}
	return s.promptSum / float64(s.promptCount)
}

func (s *groupTurnStrategy) Name() string { return groupTurnStrategyName }

func (s *groupTurnStrategy) getOrCreateState(id string) *groupTurnState {
	if a, ok := s.state.Load(id); ok {
		if st, ok := a.(*groupTurnState); ok {
			return st
		}
	}
	fresh := &groupTurnState{}
	actual, _ := s.state.LoadOrStore(id, fresh)
	if st, ok := actual.(*groupTurnState); ok {
		return st
	}
	s.state.Store(id, fresh)
	return fresh
}

type groupTurnWaiting struct {
	id      string
	qi      QueueInfo
	headEnq time.Time
}

func (s *groupTurnStrategy) Pick(_ int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor {
	now := time.Now()

	// Self-balancing controllers (slow clocks, own locks; fairness untouched).
	if s.adaptive && s.sb != nil {
		if s.sb.tickG(now) {
			s.sb.stepG(s.sbRunningCount(), s.gMax, now, s.waitingOutsideCount(queues))
			groupTurnGroupTarget.Set(float64(s.sb.effectiveG()))
		}
		if s.sb.tickTau(now) {
			s.sb.stepTau(s.kneeMax, now)
			groupTurnKneeSeconds.Set(s.sb.knee().Seconds())
		}
	}

	waiting := make([]groupTurnWaiting, 0, len(queues))
	waitingIDs := make(map[string]bool, len(queues))
	for id, qi := range queues {
		if qi.Metrics == nil || qi.Len == 0 || qi.Queue == nil {
			continue
		}
		head := qi.Queue.Peek()
		if head == nil {
			continue
		}
		waiting = append(waiting, groupTurnWaiting{id: id, qi: qi, headEnq: head.EnqueueTime()})
		waitingIDs[id] = true
	}
	if len(waiting) == 0 {
		return nil
	}

	s.groupMu.Lock()
	defer s.groupMu.Unlock()

	if s.group == nil {
		s.group = make(map[string]struct{})
	}

	// 1. Hold-time exits: members with nothing pending or in-flight for longer
	// than the hold duration leave, freeing their seat.
	for id := range s.group {
		if waitingIDs[id] {
			continue
		}
		st := s.getOrCreateState(id)
		_, _, inFlight, _, lastActivity := st.snapshot()
		if inFlight > 0 {
			continue
		}
		if now.Sub(lastActivity) > s.holdDuration {
			s.removeMemberLocked(id)
			groupTurnHoldExits.Inc()
		}
	}

	// 2-7 (seat fills, max-wait bound, member pick, all-idle victim path,
	// defensive fallback, turn-quota swap) plus the cold-write budget loop live in
	// pickWithColdBudget (strategy_groupturn_cold.go); with the budget off it
	// is the plain selection, behaviorally identical to the pre-budget
	// strategy.
	return s.pickWithColdBudget(waiting, now)
}

// selectCandidateLocked runs the group-turn selection over selectable — the
// waiting flows eligible for dispatch right now — and returns the chosen
// program ID ("" only when nothing is admissible). all carries every waiting
// flow, cold-deferred requests included: a deferred member still has work, so
// it must suppress the all-idle trigger exactly like a dispatchable one. With
// the cold budget off, selectable == all and the selection is identical to
// the pre-budget strategy.
func (s *groupTurnStrategy) selectCandidateLocked(all, selectable []groupTurnWaiting, now time.Time) string {
	// 2. Fill vacant seats with waiting non-members, in entry-policy order.
	for len(s.group) < s.effectiveGroupSize() {
		cand := s.pickEntrantLocked(selectable, now)
		if cand == "" {
			break
		}
		s.admitMemberLocked(cand, now)
	}

	// 3. Max-wait bound: any waiting session whose head has waited >= D is
	// served next, oldest head first. This deliberately preempts warmth-EDF,
	// the turn quota and the entry contract — D is the SLO the pick rule
	// must never violate, and cutting the overload tail is exactly its job.
	// It also returns before the quota-renewal branch, so an expired member
	// dispatched out of order does not renew its stay.
	if cand := s.pickMaxWaitLocked(selectable, now); cand != "" {
		return cand
	}

	// 4. Candidate: the waiting member least recently served.
	candID := s.pickLRSMemberLocked(selectable)

	// 5. All-idle trigger: no member has work, but outsiders wait. Evict a
	// victim and seat the entry-queue head. A member with an in-flight request
	// is never a victim: its turn is in progress and evicting it would push its
	// next mid-turn request back into the entry queue. When no member is
	// evictable (all in flight), the oldest waiting outsider is gap-filled
	// without membership rather than leaving capacity idle.
	if candID == "" && !s.anyMemberWaitingLocked(all) {
		// Seat only an entrant whose head the cold budget could afford to
		// dispatch (a no-op check when the budget is off): never evict a victim
		// for a request that would immediately be deferred.
		if ent := s.pickAffordableEntrantLocked(selectable, now); ent != "" {
			if victim := s.pickVictimLocked(now); victim != "" {
				s.removeMemberLocked(victim)
				groupTurnVictimEvictions.Inc()
				if ent == victim {
					// Impossible (victim had no waiting request), but never
					// re-admit the member we just evicted.
					ent = ""
				}
				if ent != "" {
					s.admitMemberLocked(ent, now)
					candID = ent
				}
			}
		}
	}

	if candID == "" {
		// Defensive: waiting flows exist but none admissible and no member
		// waiting. Fall back to the oldest eligible head so dispatch never
		// stalls. The step-5 pre-screen (pickAffordableEntrantLocked) defers
		// unaffordable entrants in place, so by the time we get here every
		// head in `selectable` may be deferred and the re-filter empty: that
		// is an idle pick (the caller handles ""), never an index out of
		// range (live crash: exp7 t5, 2026-10-18 — oldestWaitingID[0] on a
		// zero-length slice killed the EPP's dispatch loop).
		if eligible := s.coldSelectableLocked(selectable, now); len(eligible) > 0 {
			return oldestWaitingID(eligible)
		}
		return ""
	}

	// 6. Turn quota: rotate the exhausted member out for the entry-queue head
	// (only when outsiders wait); otherwise renew its stay.
	if s.quotaExhaustedLocked(candID) {
		if s.hasWaitingNonMemberLocked(selectable) {
			s.removeMemberLocked(candID)
			groupTurnSwaps.Inc()
			if ent := s.pickAffordableEntrantLocked(selectable, now); ent != "" {
				s.admitMemberLocked(ent, now)
				candID = ent
			} else {
				// Should not happen (an outsider was waiting), but never
				// strand the pick: re-admit the leaver.
				s.admitMemberLocked(candID, now)
			}
		} else {
			s.getOrCreateState(candID).BeginStay(now)
		}
	}

	return candID
}

// anyMemberWaitingLocked reports whether any group member has a request
// waiting right now — including members whose head is deferred by the cold
// budget (deferred work still suppresses the all-idle trigger, so a cold
// request can never get a warm member evicted out from under it).
func (s *groupTurnStrategy) anyMemberWaitingLocked(waiting []groupTurnWaiting) bool {
	for _, w := range waiting {
		if _, ok := s.group[w.id]; ok {
			return true
		}
	}
	return false
}

// setMembershipGauges exports the group-size and waiting-outside gauges over
// every waiting flow — cold-deferred requests included: they are waiting too.
func (s *groupTurnStrategy) setMembershipGauges(waiting []groupTurnWaiting) {
	outside := 0
	for _, w := range waiting {
		if _, ok := s.group[w.id]; !ok {
			outside++
		}
	}
	groupTurnGroupSize.Set(float64(len(s.group)))
	groupTurnWaitingOutside.Set(float64(outside))
}

// waitingQueue returns the queue accessor for a waiting program ID.
func waitingQueue(waiting []groupTurnWaiting, id string) flowcontrol.FlowQueueAccessor {
	for _, w := range waiting {
		if w.id == id {
			return w.qi.Queue
		}
	}
	return nil
}

// pickEntrantLocked returns the waiting non-member that should join next:
// least (decayed) service, or the oldest head wait; ties break to the older
// head, then the lexicographically smaller ID for determinism.
func (s *groupTurnStrategy) pickEntrantLocked(waiting []groupTurnWaiting, now time.Time) string {
	best := ""
	var bestService float64
	var bestEnq time.Time
	fallbackMean := s.globalMeanCost()
	for _, w := range waiting {
		if _, ok := s.group[w.id]; ok {
			continue
		}
		svc := s.getOrCreateState(w.id).EffectiveService(now, s.halfLifeSeconds, fallbackMean)
		if best == "" || entrantBetter(s.entryPolicy, svc, w.headEnq, w.id, bestService, bestEnq, best) {
			best, bestService, bestEnq = w.id, svc, w.headEnq
		}
	}
	return best
}

func entrantBetter(policy string, svc float64, enq time.Time, id string, bestSvc float64, bestEnq time.Time, bestID string) bool {
	if policy == groupTurnEntryOldestWait {
		if !enq.Equal(bestEnq) {
			return enq.Before(bestEnq)
		}
		return id < bestID
	}
	if svc != bestSvc {
		return svc < bestSvc
	}
	if !enq.Equal(bestEnq) {
		return enq.Before(bestEnq)
	}
	return id < bestID
}

// pickMaxWaitLocked returns the waiting session whose head request has
// waited past the max-wait bound D, oldest head first (FCFS among the
// expired; ties break to the lexicographically smaller ID for determinism).
// Any session qualifies, member or not: with the v3 sentinel group everyone
// is a member, but the rule is written generally — an expired outsider is
// dispatched without membership, exactly like the all-idle gap fill, because
// the SLO bound outranks the entry contract. maxWait <= 0 (zero-value
// strategies in tests) disables the bound.
func (s *groupTurnStrategy) pickMaxWaitLocked(waiting []groupTurnWaiting, now time.Time) string {
	if s.maxWait <= 0 {
		return ""
	}
	best := ""
	var bestEnq time.Time
	for _, w := range waiting {
		if now.Sub(w.headEnq) < s.maxWait {
			continue
		}
		if best == "" || w.headEnq.Before(bestEnq) || (w.headEnq.Equal(bestEnq) && w.id < best) {
			best, bestEnq = w.id, w.headEnq
		}
	}
	if best != "" {
		// Count only actual D-lane picks, not every check: this is the
		// live run's D-fire rate (the falsification target is "rare at
		// <=1x load, present at overload").
		groupTurnMaxWaitPicks.Inc()
	}
	return best
}

// pickLRSMemberLocked returns the waiting member to dispatch next. With
// urgencySeconds > 0 (warmth-EDF), warm members — those whose prefix is still
// inside the decay knee (now - lastDispatch < urgency) — are served first,
// earliest expiry first (the oldest warm member is closest to losing its
// cache; dispatching re-reads and refreshes it). A never-dispatched member is
// cold by definition (no prefix to save). When no member is warm, the rule
// falls back to least-recently-served order over all waiting members.
func (s *groupTurnStrategy) pickLRSMemberLocked(waiting []groupTurnWaiting) string {
	bestWarm, bestAny := "", ""
	var bestWarmDispatch, bestAnyDispatch, bestWarmEnq, bestAnyEnq time.Time
	now := time.Now()
	knee := s.warmKnee()
	useUrgency := knee > 0
	for _, w := range waiting {
		if _, ok := s.group[w.id]; !ok {
			continue
		}
		_, _, _, lastDispatch, _ := s.getOrCreateState(w.id).snapshot()
		if bestAny == "" || lastDispatch.Before(bestAnyDispatch) ||
			(lastDispatch.Equal(bestAnyDispatch) && w.headEnq.Before(bestAnyEnq)) {
			bestAny, bestAnyDispatch, bestAnyEnq = w.id, lastDispatch, w.headEnq
		}
		if useUrgency && now.Sub(lastDispatch) < knee {
			if bestWarm == "" || lastDispatch.Before(bestWarmDispatch) ||
				(lastDispatch.Equal(bestWarmDispatch) && w.headEnq.Before(bestWarmEnq)) {
				bestWarm, bestWarmDispatch, bestWarmEnq = w.id, lastDispatch, w.headEnq
			}
		}
	}
	if bestWarm != "" {
		return bestWarm
	}
	return bestAny
}

// pickVictimLocked returns the member to evict on the all-idle trigger:
// longest idle (oldest lastActivity) by default, or least-service. Members
// with an in-flight request are never eligible: their turn is in progress.
func (s *groupTurnStrategy) pickVictimLocked(now time.Time) string {
	best := ""
	var bestActivity time.Time
	var bestService float64
	for id := range s.group {
		st := s.getOrCreateState(id)
		_, _, inFlight, _, lastActivity := st.snapshot()
		if inFlight != 0 {
			continue
		}
		svc := st.Service(now, s.halfLifeSeconds)
		if best == "" {
			best, bestActivity, bestService = id, lastActivity, svc
			continue
		}
		if s.victimPolicy == groupTurnVictimLeastService {
			if svc < bestService {
				best, bestActivity, bestService = id, lastActivity, svc
			}
		} else if lastActivity.Before(bestActivity) {
			best, bestActivity, bestService = id, lastActivity, svc
		}
	}
	return best
}

func (s *groupTurnStrategy) quotaExhaustedLocked(id string) bool {
	dispatchedTotal, stayStart, _, _, _ := s.getOrCreateState(id).snapshot()
	return dispatchedTotal-stayStart >= s.turnQuota
}

func (s *groupTurnStrategy) hasWaitingNonMemberLocked(waiting []groupTurnWaiting) bool {
	for _, w := range waiting {
		if _, ok := s.group[w.id]; !ok {
			return true
		}
	}
	return false
}

func (s *groupTurnStrategy) admitMemberLocked(id string, now time.Time) {
	s.group[id] = struct{}{}
	s.getOrCreateState(id).BeginStay(now)
	if s.adaptive && s.sb != nil {
		s.sb.noteAdmission()
	}
	groupTurnAdmissions.Inc()
}

// removeMemberLocked drops group membership. Under the reset credit policy the
// program's service also restarts from zero, making it look like a newcomer on
// re-entry; under keep (default) it retains its decayed service.
func (s *groupTurnStrategy) removeMemberLocked(id string) {
	delete(s.group, id)
	if s.creditPolicy == groupTurnCreditReset {
		s.getOrCreateState(id).ResetService()
	}
}

func oldestWaitingID(waiting []groupTurnWaiting) string {
	best := waiting[0]
	for _, w := range waiting[1:] {
		if w.headEnq.Before(best.headEnq) {
			best = w
		}
	}
	return best.id
}

func (s *groupTurnStrategy) OnPreRequest(_ *ProgramMetrics, request *fwksched.InferenceRequest) {
	if request == nil {
		return
	}
	now := time.Now()
	st := s.getOrCreateState(programIDFor(request))
	st.CapturePendingGap(now)
	st.RecordDispatch(now)
}

// OnAborted settles a stream that terminated without completing (client
// disconnect, eviction, upstream error). The plugin calls it from the final
// EndOfStream callback when the framework reports a non-natural termination
// cause; without it, a disconnected client would leak the in-flight seat
// forever — blocking hold-exit and TTL eviction and pinning the pending
// service term in effective service (p14 §9.1).
func (s *groupTurnStrategy) OnAborted(_ *ProgramMetrics, request *fwksched.InferenceRequest) {
	if request == nil {
		return
	}
	s.getOrCreateState(programIDFor(request)).RecordAbort(time.Now())
}

func (s *groupTurnStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	id := programIDFor(request)
	st := s.getOrCreateState(id)
	now := time.Now()
	promptTokens := float64(response.Usage.PromptTokens)
	st.RecordCompletion(now, promptTokens)
	cost := float64(weightInputToken*int64(response.Usage.PromptTokens) + weightOutputToken*int64(response.Usage.CompletionTokens))
	st.AddService(cost, now, s.halfLifeSeconds)
	s.pendingMu.Lock()
	s.pendingSum += cost
	s.pendingCount++
	s.promptSum += promptTokens
	s.promptCount++
	s.pendingMu.Unlock()
	if s.adaptive && s.sb != nil {
		if gap := st.takePendingGap(); gap > 0 && response.Usage.PromptTokenDetails != nil {
			s.sb.feedReEntry(gap, float64(response.Usage.PromptTokenDetails.CachedTokens), float64(response.Usage.PromptTokens), now)
		}
	}
	groupTurnServiceTokens.WithLabelValues(id).Set(st.Service(now, s.halfLifeSeconds))
}

func (s *groupTurnStrategy) EvictProgram(id string) {
	s.state.Delete(id)
	s.groupMu.Lock()
	if s.group != nil {
		delete(s.group, id)
	}
	if s.coldDeferUntil != nil {
		delete(s.coldDeferUntil, id)
	}
	s.groupMu.Unlock()
	groupTurnServiceTokens.DeleteLabelValues(id)
}

func (s *groupTurnStrategy) Collectors() []prometheus.Collector {
	return []prometheus.Collector{
		groupTurnGroupSize,
		groupTurnWaitingOutside,
		groupTurnAdmissions,
		groupTurnSwaps,
		groupTurnVictimEvictions,
		groupTurnHoldExits,
		groupTurnMaxWaitPicks,
		groupTurnServiceTokens,
		// Self-balancing controller gauges (defined in
		// strategy_groupturn_sb.go; previously never registered, so they
		// silently did not export).
		groupTurnGroupTarget,
		groupTurnKneeSeconds,
		// Cold-write budget (strategy_groupturn_cold.go).
		groupTurnColdCredit,
		groupTurnColdChargedTokens,
		groupTurnColdDefers,
		groupTurnColdIdlePicks,
		groupTurnColdNoPrediction,
	}
}
