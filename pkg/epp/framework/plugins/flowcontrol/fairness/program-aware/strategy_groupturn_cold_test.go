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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newColdGroupTurn builds a budget-enabled strategy with a stubbed charge
// estimator (the production Pick-time prediction source is unwired — see
// strategy_groupturn_cold.go).
func newColdGroupTurn(groupSize int, quota int64, budget float64, tenants int, charge coldChargeEstimator) *groupTurnStrategy {
	s := newGroupTurn(groupSize, quota)
	s.coldBudgetTokensPerSec = budget
	s.coldBudgetTenants = tenants
	s.coldCharge = charge
	return s
}

// makeInfoReq is makeInfo with a head that carries an InferenceRequest (the
// charge estimator's input).
func makeInfoReq(id string, headEnqueue time.Time) (string, QueueInfo) {
	item := &fwkfcmocks.MockQueueItemAccessor{
		EnqueueTimeV: headEnqueue,
		OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{
			FlowKeyV:          flowcontrol.FlowKey{ID: id},
			InferenceRequestV: &fwksched.InferenceRequest{FairnessID: id},
		},
	}
	q := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: id},
		PeekV:    item,
	}
	return id, QueueInfo{Queue: q, Metrics: &ProgramMetrics{}, Len: 1}
}

// chargeByFairnessID stubs the estimator: cold tokens per FairnessID, 0 for
// everyone else, ok always.
func chargeByFairnessID(cold map[string]float64) coldChargeEstimator {
	return func(request *fwksched.InferenceRequest) (float64, bool) {
		if request == nil {
			return 0, false
		}
		return cold[request.FairnessID], true
	}
}

// sameTenantPair returns two program IDs that hash to the same budget tenant.
func sameTenantPair(s *groupTurnStrategy) (string, string) {
	for i := 1; ; i++ {
		other := fmt.Sprintf("prog-%d", i)
		if s.coldTenantOf(other) == s.coldTenantOf("prog-0") {
			return "prog-0", other
		}
	}
}

// drainColdBucket pins a tenant's bucket credit low so charges become
// unaffordable (buckets start full — debt semantics dispatch an oversized
// charge on a full bucket — so tests that need deferral drain first, the
// state a bucket is in during the cold carousel).
func drainColdBucket(s *groupTurnStrategy, id string, credit float64, now time.Time) {
	s.groupMu.Lock()
	defer s.groupMu.Unlock()
	s.coldBucketLocked(s.coldTenantOf(id), now).credit = credit
}

func TestGroupTurnCold_BudgetOffIsNoOp(t *testing.T) {
	called := false
	s := newColdGroupTurn(4, 4, 0, 2, func(*fwksched.InferenceRequest) (float64, bool) {
		called = true
		return 1e9, true
	})
	now := time.Now()
	idA, qA := makeInfoReq("alpha", now.Add(-1*time.Second))
	idB, qB := makeInfoReq("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID, "budget 0: baseline dispatch behavior")
	assert.False(t, called, "estimator never consulted with the budget off")
	assert.Empty(t, s.coldDeferUntil, "no deferrals recorded with the budget off")
	assert.Nil(t, s.coldBuckets, "no buckets created with the budget off")
}

func TestGroupTurnCold_TenantHashStable(t *testing.T) {
	s := newColdGroupTurn(4, 4, 9000, 2, chargeByFairnessID(nil))
	other := newColdGroupTurn(4, 4, 9000, 2, chargeByFairnessID(nil))

	for _, id := range []string{"alpha", "beta", "session-1234", ""} {
		assert.Equal(t, s.coldTenantOf(id), other.coldTenantOf(id), "tenant derivation is restart-stable for %q", id)
		assert.GreaterOrEqual(t, s.coldTenantOf(id), 0)
		assert.Less(t, s.coldTenantOf(id), 2)
	}
	// The split covers both tenants over a session-like population.
	seen := map[int]bool{}
	for i := 0; i < 200; i++ {
		seen[s.coldTenantOf(fmt.Sprintf("session-%d", i))] = true
	}
	assert.True(t, seen[0] && seen[1], "both tenants populated (parity-split analog)")

	s3 := newColdGroupTurn(4, 4, 9000, 3, chargeByFairnessID(nil))
	seen3 := map[int]bool{}
	for i := 0; i < 300; i++ {
		seen3[s3.coldTenantOf(fmt.Sprintf("session-%d", i))] = true
	}
	assert.True(t, seen3[0] && seen3[1] && seen3[2], "tenant count is configurable")
}

func TestGroupTurnCold_BucketDebtAndRefill(t *testing.T) {
	// Total 200 tok/s over 2 tenants: rate 100/tenant, capacity 1000 (10 s window).
	s := newColdGroupTurn(4, 4, 200, 2, chargeByFairnessID(nil))
	now := time.Now()
	assert.InDelta(t, 100.0, s.coldRateLocked(), 1e-9)
	assert.InDelta(t, 1000.0, s.coldCapacityLocked(), 1e-9)

	// New buckets start full; an oversized charge (5000 > capacity) is
	// dispatchable at full credit and drives the bucket into debt.
	b := s.coldBucketLocked(0, now)
	assert.InDelta(t, 1000.0, b.credit, 1e-9)
	assert.True(t, s.coldTryChargeLocked(0, 5000, now), "debt semantics: oversized charge affordable on a full bucket")
	assert.InDelta(t, -4000.0, b.credit, 1e-9, "credit goes negative (debt)")

	// While in debt even a small charge is unaffordable and leaves the bucket untouched.
	assert.False(t, s.coldTryChargeLocked(0, 10, now))
	assert.InDelta(t, -4000.0, b.credit, 1e-9)

	// Refill accrues at the rate and is capped at capacity.
	later := now.Add(10 * time.Second)
	s.coldRefillLocked(later)
	assert.InDelta(t, -4000.0+1000.0, b.credit, 1e-9, "debt refills at exactly the configured rate")
	s.coldRefillLocked(later.Add(100 * time.Second))
	assert.InDelta(t, 1000.0, b.credit, 1e-9, "refill caps at capacity (rate x window)")

	// A charge below capacity is affordable only once credit reaches it.
	small := s.coldBucketLocked(1, now)
	small.credit = 200
	assert.False(t, s.coldTryChargeLocked(1, 500, now))
	assert.InDelta(t, 200.0, small.credit, 1e-9)
	small.credit = 500
	assert.True(t, s.coldTryChargeLocked(1, 500, now))
	assert.InDelta(t, 0.0, small.credit, 1e-9)

	// The deferral need is (threshold - credit)/rate, floored at 50 ms.
	small.credit = 100
	s.coldDeferLocked("some-prog", 500, now) // tenant of "some-prog" irrelevant to the math below
	// Recompute for tenant 1 explicitly: threshold 500, credit 100, rate 100 -> 4 s.
	need := time.Duration((coldThreshold(500, s.coldCapacityLocked()) - 100) / s.coldRateLocked() * float64(time.Second))
	assert.InDelta(t, 4.0, need.Seconds(), 1e-9)
}

func TestGroupTurnCold_PerRequestDeferralNeverBlocksWarm(t *testing.T) {
	// Two programs of the SAME tenant: one cold-unaffordable head, one warm.
	// The p7b finding ported: deferral must never mask the tenant's warm
	// dispatches.
	s := newColdGroupTurn(4, 100, 200, 2, chargeByFairnessID(nil))
	coldID, warmID := sameTenantPair(s)
	require.Equal(t, s.coldTenantOf(coldID), s.coldTenantOf(warmID), "test premise: same tenant")
	cold := map[string]float64{coldID: 5000} // >> capacity 1000

	s.coldCharge = chargeByFairnessID(cold)
	now := time.Now()
	idC, qC := makeInfoReq(coldID, now.Add(-2*time.Second))
	idW, qW := makeInfoReq(warmID, now.Add(-1*time.Second))
	queues := map[string]QueueInfo{idC: qC, idW: qW}
	drainColdBucket(s, coldID, 100, now) // same tenant as warmID: unaffordable

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, warmID, got.FlowKey().ID, "warm head of the SAME tenant dispatches while the cold one is deferred")
	assert.True(t, s.isMember(coldID), "deferred program keeps membership (deferred != evicted)")
	until, deferred := s.coldDeferUntil[coldID]
	assert.True(t, deferred, "cold request deferred per-request")
	assert.True(t, until.After(now), "deferral has a future deadline (until the bucket could afford the charge)")

	// The cold program alone: honest idle, then re-entry after refill.
	queuesCold := map[string]QueueInfo{idC: qC}
	assert.Nil(t, s.Pick(0, queuesCold), "nothing affordable: Pick returns nil (binding-budget idle)")
	assert.Nil(t, s.Pick(0, queuesCold), "still deferred on the immediate next Pick")

	// Bucket refilled to full (capacity) and deferral expired: the oversized
	// charge dispatches on the full bucket — debt semantics — and drives
	// credit negative.
	s.groupMu.Lock()
	s.coldDeferUntil[coldID] = now.Add(-time.Millisecond)
	s.coldBucketLocked(s.coldTenantOf(coldID), now).credit = s.coldCapacityLocked()
	s.groupMu.Unlock()
	got = s.Pick(0, queuesCold)
	require.NotNil(t, got)
	assert.Equal(t, coldID, got.FlowKey().ID, "deferred request re-enters ranking once the bucket refills")
	tenant := s.coldTenantOf(coldID)
	assert.InDelta(t, s.coldCapacityLocked()-5000, s.coldBuckets[tenant].credit, 1e-9, "oversized dispatch on a full bucket charges into debt")
}

func TestGroupTurnCold_DeferredMemberSuppressesAllIdleEviction(t *testing.T) {
	// A deferred member still has work: the all-idle trigger must not evict
	// anyone to seat another (possibly unaffordable) outsider.
	s := newColdGroupTurn(1, 100, 200, 2, chargeByFairnessID(nil))
	now := time.Now()
	idA, qA := makeInfoReq("alpha", now)
	queues := map[string]QueueInfo{idA: qA}
	require.NotNil(t, s.Pick(0, queues))
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10, 5) // member idle, in-flight 0

	// Same-tenant outsider with a cold-unaffordable head; alpha's own head is
	// also cold (the deferred-member scenario): both queue. Drain both
	// tenants' buckets so neither charge is affordable.
	_, partner := sameTenantPair(s)
	cold := map[string]float64{"alpha": 5000, partner: 5000}
	s.coldCharge = chargeByFairnessID(cold)
	drainColdBucket(s, "alpha", 100, now)
	drainColdBucket(s, partner, 100, now)
	idP, qP := makeInfoReq(partner, now)
	queues2 := map[string]QueueInfo{idA: qA, idP: qP}

	got := s.Pick(0, queues2)
	assert.Nil(t, got, "everything cold and unaffordable: nil")
	assert.True(t, s.isMember("alpha"), "deferred member with waiting work is never all-idle-evicted")
	_, deferred := s.coldDeferUntil[partner]
	assert.True(t, deferred, "the cold outsider was deferred during entrant pre-screening")
}

func TestGroupTurnCold_QuotaSwapSkipsUnaffordableEntrant(t *testing.T) {
	// A quota-exhausted member with a WARM head must not be rotated out for
	// an entrant whose cold head would be deferred: the pre-screened swap
	// falls through to re-admitting the leaver.
	s := newColdGroupTurn(1, 1, 200, 2, chargeByFairnessID(nil))
	now := time.Now()
	idA, qA := makeInfoReq("alpha", now.Add(-2*time.Second))
	queues := map[string]QueueInfo{idA: qA}
	require.NotNil(t, s.Pick(0, queues))
	gtDispatch(s, "alpha") // quota of 1 exhausted; alpha's head is warm (charge 0)

	_, partner := sameTenantPair(s)
	s.coldCharge = chargeByFairnessID(map[string]float64{partner: 5000})
	drainColdBucket(s, partner, 100, now) // unaffordable: threshold 1000 > credit 100
	idB, qB := makeInfoReq(partner, now)
	queues2 := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues2)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID, "leaver re-admitted and dispatched when no affordable entrant exists")
	assert.True(t, s.isMember("alpha"), "leaver keeps its seat")
	assert.False(t, s.isMember(partner), "unaffordable entrant is not seated")
	_, deferred := s.coldDeferUntil[partner]
	assert.True(t, deferred, "unaffordable entrant deferred during the swap pre-screen")
}

func TestGroupTurnCold_AllDeferredFallbackIsIdleNotPanic(t *testing.T) {
	// Live crash (exp7 t5, 2026-10-18): the group is full of members with no
	// waiting work; every waiting head is a non-member whose cold charge is
	// unaffordable. selectCandidateLocked's step-4 pre-screen
	// (pickAffordableEntrantLocked) defers each head while scanning entrants,
	// so the defensive fallback's re-filter of `selectable` came back EMPTY —
	// and oldestWaitingID indexed [0] of a zero-length slice, crashing the
	// EPP's dispatch loop (panic every few minutes under c256 load). The
	// fallback must return "" (the caller's idle-pick path), never panic.
	s := newColdGroupTurn(1, 100, 100, 2, chargeByFairnessID(nil))
	now := time.Now()

	// Seat and idle the lone member: the group is then full of non-waiting
	// members (step 2 finds no vacancy, step 3 no waiting member), which is
	// the deep-phase steady state that produced the live crash.
	idA, qA := makeInfoReq("alpha", now)
	require.NotNil(t, s.Pick(0, map[string]QueueInfo{idA: qA}))
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10, 5)

	// Two outsiders with oversized cold charges and drained buckets: the
	// entrant pre-screen defers both while scanning.
	idB, qB := makeInfoReq("bravo", now.Add(-time.Second))
	idC, qC := makeInfoReq("charlie", now.Add(-2*time.Second))
	s.coldCharge = chargeByFairnessID(map[string]float64{"bravo": 5000, "charlie": 5000})
	drainColdBucket(s, "bravo", 100, now)
	drainColdBucket(s, "charlie", 100, now)

	got := s.Pick(0, map[string]QueueInfo{idB: qB, idC: qC})
	assert.Nil(t, got, "every head deferred by the pre-screen: idle pick (nil), not a panic")
	_, defB := s.coldDeferUntil["bravo"]
	_, defC := s.coldDeferUntil["charlie"]
	assert.True(t, defB && defC, "both outsider heads were deferred during entrant pre-screening")
}

func TestGroupTurnCold_ChargePathWithStub(t *testing.T) {
	// Affordable predicted cold: charged at the dispatch decision.
	s := newColdGroupTurn(4, 100, 2000, 2, chargeByFairnessID(map[string]float64{"alpha": 1234}))
	now := time.Now()
	idA, qA := makeInfoReq("alpha", now)
	queues := map[string]QueueInfo{idA: qA}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID)
	tenant := s.coldTenantOf("alpha")
	capacity := s.coldCapacityLocked()
	assert.InDelta(t, capacity-1234, s.coldBuckets[tenant].credit, 1e-9, "bucket charged the stubbed prediction")

	// ok=false (no prediction): fail-open, treated as warm.
	s2 := newColdGroupTurn(4, 100, 2000, 2, func(*fwksched.InferenceRequest) (float64, bool) {
		return 99999, false
	})
	idB, qB := makeInfoReq("beta", now)
	got = s2.Pick(0, map[string]QueueInfo{idB: qB})
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID, "missing prediction never gates dispatch")
	assert.Empty(t, s2.coldBuckets, "no bucket is charged without a prediction")
	assert.Empty(t, s2.coldDeferUntil)

	// Nil estimator (direct-construction misuse; config validation refuses
	// this state): fail-open rather than wedge dispatch.
	s3 := newColdGroupTurn(4, 100, 2000, 2, nil)
	got = s3.Pick(0, map[string]QueueInfo{idB: qB})
	require.NotNil(t, got)
	assert.Empty(t, s3.coldDeferUntil)
}

func TestGroupTurnCold_WarmChargeZeroNeverDeferred(t *testing.T) {
	// The ratchet: predicted cold 0 charges nothing and is always affordable,
	// even with the budget tiny and the bucket empty.
	s := newColdGroupTurn(4, 100, 0.001, 2, chargeByFairnessID(nil)) // capacity ~0.005 tokens
	now := time.Now()
	for i := 0; i < 20; i++ {
		id, q := makeInfoReq(fmt.Sprintf("warm-%d", i), now)
		got := s.Pick(0, map[string]QueueInfo{id: q})
		require.NotNil(t, got, "warm dispatch %d never gated", i)
	}
	assert.Empty(t, s.coldDeferUntil)
}

func TestGroupTurnCold_EvictProgramClearsDeferral(t *testing.T) {
	s := newColdGroupTurn(4, 100, 200, 2, chargeByFairnessID(map[string]float64{"alpha": 500}))
	now := time.Now()
	drainColdBucket(s, "alpha", 100, now) // charge 500 > threshold-drained credit: deferred
	idA, qA := makeInfoReq("alpha", now)
	assert.Nil(t, s.Pick(0, map[string]QueueInfo{idA: qA}))
	assert.NotEmpty(t, s.coldDeferUntil)
	s.EvictProgram("alpha")
	assert.Empty(t, s.coldDeferUntil, "eviction clears the deferral record")
}

func TestGroupTurnCold_ConfigValidation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	require.NoError(t, cfg.validate(), "default (budget off) validates")

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnColdBudgetTokensPerSec = -1
	assert.Error(t, cfg.validate(), "negative budget refused")

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnColdBudgetTokensPerSec = 9000
	cfg.GroupTurnColdBudgetTenants = 0
	err := cfg.validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "groupTurnColdBudgetTenants", "tenant sanity checked ahead of the blocker")

	// The knee requirement: the estimator classifies warmth by the warmth
	// knee, so a budget without one degenerates to prompt denomination (the
	// p7 idleness failure mode) and is refused. Urgency is zeroed explicitly:
	// v3 flipped its default to 120 (fixed knee), and this case pins the
	// no-knee refusal.
	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnColdBudgetTokensPerSec = 9000
	cfg.GroupTurnUrgencySeconds = 0
	err = cfg.validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "warmth knee", "budget without a knee refused")

	// Budget on with a knee (fixed or adaptive): valid.
	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnColdBudgetTokensPerSec = 9000
	cfg.GroupTurnUrgencySeconds = 60
	require.NoError(t, cfg.validate())

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnColdBudgetTokensPerSec = 9000
	cfg.GroupTurnAdaptive = true
	require.NoError(t, cfg.validate(), "adaptive provides the knee (sb bootstrap 60s, clamped [30, kneeMax])")
}

func TestGroupTurnCold_Wiring(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	st, err := newStrategy(cfg)
	require.NoError(t, err)
	gt, ok := st.(*groupTurnStrategy)
	require.True(t, ok)
	assert.Zero(t, gt.coldBudgetTokensPerSec, "budget off by default")
	assert.Equal(t, 2, gt.coldBudgetTenants)
	assert.NotNil(t, gt.coldCharge, "knee-keyed estimator wired even with the budget off (inert until enabled)")

	cfg.GroupTurnColdBudgetTokensPerSec = 9000
	cfg.GroupTurnColdBudgetTenants = 4
	st, err = newStrategy(cfg)
	require.NoError(t, err)
	gt, ok = st.(*groupTurnStrategy)
	require.True(t, ok)
	assert.Equal(t, 9000.0, gt.coldBudgetTokensPerSec)
	assert.Equal(t, 4, gt.coldBudgetTenants)
	assert.NotNil(t, gt.coldCharge, "knee-keyed estimator wired as the production charge source")
}

func newKneeKeyed(groupSize int, quota int64, budget float64) *groupTurnStrategy {
	s := newGroupTurn(groupSize, quota)
	s.coldBudgetTokensPerSec = budget
	s.coldBudgetTenants = 2
	s.urgencySeconds = 60 * time.Second
	s.coldCharge = s.kneeKeyedColdCharge
	return s
}

func TestGroupTurnCold_KneeKeyed_WarmNeverCharged(t *testing.T) {
	s := newKneeKeyed(4, 100, 9000)
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10000, 500) // lastPrompt = 10000
	st := s.getOrCreateState("alpha")
	st.mu.Lock()
	st.lastDispatch = time.Now().Add(-30 * time.Second) // inside the 60 s knee
	st.mu.Unlock()

	charge, ok := s.coldCharge(&fwksched.InferenceRequest{FairnessID: "alpha"})
	require.True(t, ok)
	assert.Zero(t, charge, "warm session (gap inside the knee) charges zero — the ratchet")
}

func TestGroupTurnCold_KneeKeyed_ColdChargesLastPrompt(t *testing.T) {
	s := newKneeKeyed(4, 100, 9000)
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10000, 500)
	st := s.getOrCreateState("alpha")
	st.mu.Lock()
	st.lastDispatch = time.Now().Add(-120 * time.Second) // beyond the knee
	st.mu.Unlock()

	charge, ok := s.coldCharge(&fwksched.InferenceRequest{FairnessID: "alpha"})
	require.True(t, ok)
	assert.InDelta(t, 10000.0, charge, 1e-9, "cold re-entry charges the last-known prompt size (the re-prefill scale)")

	// Freshly-created state (dispatched but never completed, and past the
	// knee): falls to the global mean, not zero.
	gtDispatch(s, "beta")
	bst := s.getOrCreateState("beta")
	bst.mu.Lock()
	bst.lastDispatch = time.Now().Add(-120 * time.Second)
	bst.mu.Unlock()
	mean := s.globalMeanPrompt()
	charge, ok = s.coldCharge(&fwksched.InferenceRequest{FairnessID: "beta"})
	require.True(t, ok)
	assert.InDelta(t, mean, charge, 1e-9, "no completion history: the global mean prompt")
}

func TestGroupTurnCold_KneeKeyed_NewcomerGlobalMean(t *testing.T) {
	s := newKneeKeyed(4, 100, 9000)
	// No completions anywhere yet: the bootstrap constant.
	charge, ok := s.coldCharge(&fwksched.InferenceRequest{FairnessID: "new-session"})
	require.True(t, ok)
	assert.InDelta(t, groupTurnDefaultMeanPrompt, charge, 1e-9)

	// After completions land, the running mean takes over.
	for i := 0; i < 10; i++ {
		gtComplete(s, fmt.Sprintf("sess-%d", i), 2000, 10)
	}
	charge, ok = s.coldCharge(&fwksched.InferenceRequest{FairnessID: "another-new-session"})
	require.True(t, ok)
	assert.InDelta(t, 2000.0, charge, 1e-9, "newcomer charge tracks the global running mean prompt")
}

func TestGroupTurnCold_KneeKeyed_AdaptiveKneeInteraction(t *testing.T) {
	// Adaptive: the estimator must use the CONTROLLER's knee, not the
	// configured urgency bootstrap.
	s := newAdaptive(256)
	s.coldBudgetTokensPerSec = 9000
	s.coldBudgetTenants = 2
	s.coldCharge = s.kneeKeyedColdCharge
	s.sb.mu.Lock()
	s.sb.currentKnee = 120 * time.Second
	s.sb.mu.Unlock()

	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 8000, 100)
	st := s.getOrCreateState("alpha")
	st.mu.Lock()
	st.lastDispatch = time.Now().Add(-90 * time.Second) // beyond 60 s, inside 120 s
	st.mu.Unlock()

	charge, ok := s.coldCharge(&fwksched.InferenceRequest{FairnessID: "alpha"})
	require.True(t, ok)
	assert.Zero(t, charge, "adaptive knee 120 s: a 90 s gap is warm")

	// The same gap under the fixed 60 s knee is cold.
	fixed := newKneeKeyed(4, 100, 9000)
	gtDispatch(fixed, "alpha")
	gtComplete(fixed, "alpha", 8000, 100)
	fst := fixed.getOrCreateState("alpha")
	fst.mu.Lock()
	fst.lastDispatch = time.Now().Add(-90 * time.Second)
	fst.mu.Unlock()
	charge, ok = fixed.coldCharge(&fwksched.InferenceRequest{FairnessID: "alpha"})
	require.True(t, ok)
	assert.InDelta(t, 8000.0, charge, 1e-9, "fixed 60 s knee: the same 90 s gap is cold")
}

func TestGroupTurnCold_KneeKeyed_PickEndToEnd(t *testing.T) {
	// The default estimator through the full Pick path: a warm session
	// dispatches freely; a cold re-entry with an unaffordable last-prompt
	// defers; after the bucket refills it dispatches and is charged.
	s := newKneeKeyed(4, 100, 200) // 100/tenant, capacity 1000
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 500, 10) // lastPrompt = 500 (< capacity)
	st := s.getOrCreateState("alpha")
	st.mu.Lock()
	st.lastDispatch = time.Now().Add(-120 * time.Second) // cold re-entry
	st.mu.Unlock()

	idA, qA := makeInfoReq("alpha", time.Now())
	queues := map[string]QueueInfo{idA: qA}

	// First encounter: the bucket starts full, so alpha's 500-token cold
	// charge is affordable (debt semantics would allow even larger).
	assert.NotNil(t, s.Pick(0, queues), "first encounter dispatches on the full bucket")

	gtComplete(s, "alpha", 500, 10)
	st.mu.Lock()
	st.lastDispatch = time.Now().Add(-120 * time.Second)
	st.mu.Unlock()
	drainColdBucket(s, "alpha", 100, time.Now())
	assert.Nil(t, s.Pick(0, queues), "cold re-entry with a drained bucket is deferred")
	_, deferred := s.coldDeferUntil["alpha"]
	assert.True(t, deferred)

	// Warm gap: never deferred, whatever the bucket.
	st.mu.Lock()
	st.lastDispatch = time.Now().Add(-10 * time.Second)
	st.mu.Unlock()
	tenant := s.coldTenantOf("alpha")
	s.groupMu.Lock()
	delete(s.coldDeferUntil, "alpha")
	s.coldBucketLocked(tenant, time.Now()).credit = 0
	s.groupMu.Unlock()
	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID, "warm gap dispatches even with an empty bucket — the ratchet")
}
