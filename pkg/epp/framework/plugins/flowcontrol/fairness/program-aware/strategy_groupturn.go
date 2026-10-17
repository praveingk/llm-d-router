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
	inFlight        int64
	pendingGap      time.Duration // inter-dispatch gap captured at dispatch (τ estimator input)
}

func (st *groupTurnState) Service(now time.Time, halfLifeSeconds float64) float64 {
	st.mu.Lock()
	defer st.mu.Unlock()
	if halfLifeSeconds > 0 && !st.decayAnchor.IsZero() {
		if elapsed := now.Sub(st.decayAnchor).Seconds(); elapsed > 0 {
			st.service *= math.Pow(0.5, elapsed/halfLifeSeconds)
		}
	}
	st.decayAnchor = now
	return st.service
}

func (st *groupTurnState) AddService(cost float64) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.service += cost
	st.completions++
	st.meanCost += (cost - st.meanCost) / float64(st.completions)
	st.decayAnchor = time.Now()
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
	if halfLifeSeconds > 0 && !st.decayAnchor.IsZero() {
		if elapsed := now.Sub(st.decayAnchor).Seconds(); elapsed > 0 {
			st.service *= math.Pow(0.5, elapsed/halfLifeSeconds)
		}
	}
	st.decayAnchor = now
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

func (st *groupTurnState) RecordCompletion(now time.Time) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.inFlight--
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

// groupTurnStrategy schedules turns, not requests. A bounded group of programs
// holds dispatch priority — the KV-warm resident set — and membership rotates:
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
type groupTurnStrategy struct {
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

	// Self-balancing layer (strategy_groupturn_sb.go): when adaptive, the
	// group size tracks the engine's running set and the warmth knee tracks
	// the measured re-entry hit curve. Fairness is never adaptive.
	adaptive bool
	gMax     int
	kneeMax  time.Duration
	sb       *sbState

	state sync.Map // program ID (string) -> *groupTurnState

	groupMu sync.Mutex
	group   map[string]struct{} // member IDs; guarded by groupMu

	// Global running mean of completion costs; bootstraps the pending-service
	// estimate for programs with no completion history.
	pendingMu    sync.Mutex
	pendingSum   float64
	pendingCount int64
}

// effectiveGroupSize returns the current membership bound (controller-driven
// when adaptive, the configured size otherwise).
func (s *groupTurnStrategy) effectiveGroupSize() int {
	if s.adaptive && s.sb != nil {
		return s.sb.effectiveG()
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

func (s *groupTurnStrategy) globalMeanCost() float64 {
	s.pendingMu.Lock()
	defer s.pendingMu.Unlock()
	if s.pendingCount == 0 {
		return groupTurnDefaultMeanCost
	}
	return s.pendingSum / float64(s.pendingCount)
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
			s.sb.stepG(s.sbRunningCount(), s.gMax, now)
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

	// 2. Fill vacant seats with waiting non-members, in entry-policy order.
	for len(s.group) < s.effectiveGroupSize() {
		cand := s.pickEntrantLocked(waiting, now)
		if cand == "" {
			break
		}
		s.admitMemberLocked(cand, now)
	}

	// 3. Candidate: the waiting member least recently served.
	candID := s.pickLRSMemberLocked(waiting)

	// 4. All-idle trigger: no member has work, but outsiders wait. Evict a
	// victim and seat the entry-queue head. A member with an in-flight request
	// is never a victim: its turn is in progress and evicting it would push its
	// next mid-turn request back into the entry queue. When no member is
	// evictable (all in flight), the oldest waiting outsider is gap-filled
	// without membership rather than leaving capacity idle.
	if candID == "" {
		if victim := s.pickVictimLocked(now); victim != "" {
			s.removeMemberLocked(victim)
			groupTurnVictimEvictions.Inc()
			if ent := s.pickEntrantLocked(waiting, now); ent != "" {
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
		// waiting. Fall back to oldest head so dispatch never stalls.
		return oldestWaitingQueue(waiting)
	}

	// 5. Turn quota: rotate the exhausted member out for the entry-queue head
	// (only when outsiders wait); otherwise renew its stay.
	if s.quotaExhaustedLocked(candID) {
		if s.hasWaitingNonMemberLocked(waiting) {
			s.removeMemberLocked(candID)
			groupTurnSwaps.Inc()
			if ent := s.pickEntrantLocked(waiting, now); ent != "" {
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

	outside := 0
	for _, w := range waiting {
		if _, ok := s.group[w.id]; !ok {
			outside++
		}
	}
	groupTurnGroupSize.Set(float64(len(s.group)))
	groupTurnWaitingOutside.Set(float64(outside))

	return queues[candID].Queue
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

func oldestWaitingQueue(waiting []groupTurnWaiting) flowcontrol.FlowQueueAccessor {
	best := waiting[0]
	for _, w := range waiting[1:] {
		if w.headEnq.Before(best.headEnq) {
			best = w
		}
	}
	return best.qi.Queue
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

func (s *groupTurnStrategy) OnCompleted(_ *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response) {
	if request == nil || response == nil {
		return
	}
	id := programIDFor(request)
	st := s.getOrCreateState(id)
	now := time.Now()
	st.RecordCompletion(now)
	cost := float64(weightInputToken*int64(response.Usage.PromptTokens) + weightOutputToken*int64(response.Usage.CompletionTokens))
	st.AddService(cost)
	s.pendingMu.Lock()
	s.pendingSum += cost
	s.pendingCount++
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
		groupTurnServiceTokens,
	}
}
