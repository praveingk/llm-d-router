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

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func newGroupTurn(groupSize int, quota int64) *groupTurnStrategy {
	return &groupTurnStrategy{
		groupSize:       groupSize,
		turnQuota:       quota,
		entryPolicy:     groupTurnEntryLeastService,
		victimPolicy:    groupTurnVictimLongestIdle,
		creditPolicy:    groupTurnCreditKeep,
		holdDuration:    10 * time.Minute,
		halfLifeSeconds: 0,
	}
}

func (s *groupTurnStrategy) isMember(id string) bool {
	s.groupMu.Lock()
	defer s.groupMu.Unlock()
	_, ok := s.group[id]
	return ok
}

func gtDispatch(s *groupTurnStrategy, id string) {
	s.OnPreRequest(nil, &fwksched.InferenceRequest{FairnessID: id})
}

func gtComplete(s *groupTurnStrategy, id string, prompt, completion int64) {
	req := &fwksched.InferenceRequest{FairnessID: id}
	resp := &fwkrc.Response{EndOfStream: true}
	resp.Usage.PromptTokens = int(prompt)
	resp.Usage.CompletionTokens = int(completion)
	s.OnCompleted(nil, req, resp)
}

func TestGroupTurn_Name(t *testing.T) {
	assert.Equal(t, "group-turn", newGroupTurn(4, 4).Name())
}

func TestGroupTurn_CalmAllFit_EveryoneAdmitted(t *testing.T) {
	s := newGroupTurn(4, 4)
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-1*time.Second))
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID, "fresh members tie-break to older head")
	assert.True(t, s.isMember("alpha"), "both fit the group")
	assert.True(t, s.isMember("beta"))
}

func TestGroupTurn_QuotaSwap_SeatsWaitingOutsider(t *testing.T) {
	s := newGroupTurn(1, 2)
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID)
	assert.True(t, s.isMember("alpha"))
	assert.False(t, s.isMember("beta"), "group of 1 is full")

	// alpha dispatches its full turn quota of 2.
	gtDispatch(s, "alpha")
	gtDispatch(s, "alpha")

	// Next pick: alpha exhausted its stay, beta waits -> swap.
	got = s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID, "exhausted member rotates out for the entrant")
	assert.False(t, s.isMember("alpha"))
	assert.True(t, s.isMember("beta"))
}

func TestGroupTurn_QuotaRenewWhenNoOutsider(t *testing.T) {
	s := newGroupTurn(2, 1)
	now := time.Now()

	idA, qA := makeInfo("alpha", now)
	queues := map[string]QueueInfo{idA: qA}

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID)

	gtDispatch(s, "alpha") // quota of 1 exhausted

	// No outsider waits: the stay renews and alpha keeps dispatching.
	got = s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID)
	assert.True(t, s.isMember("alpha"))
}

func TestGroupTurn_AllIdleVictimEviction(t *testing.T) {
	s := newGroupTurn(1, 100)
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	queues := map[string]QueueInfo{idA: qA}
	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID)
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10, 5) // idle now, in-flight 0

	// alpha's queue is gone (tool call); beta arrives.
	idB, qB := makeInfo("beta", now)
	queues2 := map[string]QueueInfo{idB: qB}

	got = s.Pick(0, queues2)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID, "idle member evicted for the waiting outsider")
	assert.False(t, s.isMember("alpha"))
	assert.True(t, s.isMember("beta"))
}

func TestGroupTurn_HoldTimeExit(t *testing.T) {
	s := newGroupTurn(1, 100)
	s.holdDuration = 50 * time.Millisecond
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	queues := map[string]QueueInfo{idA: qA}
	require.NotNil(t, s.Pick(0, queues))
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10, 5)

	// Backdate alpha's activity past the hold time.
	st := s.getOrCreateState("alpha")
	st.mu.Lock()
	st.lastActivity = now.Add(-time.Minute)
	st.mu.Unlock()

	idB, qB := makeInfo("beta", now)
	queues2 := map[string]QueueInfo{idB: qB}
	got := s.Pick(0, queues2)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID, "seat freed by hold-time exit")
	assert.False(t, s.isMember("alpha"))
}

func TestGroupTurn_EntryLeastService(t *testing.T) {
	s := newGroupTurn(1, 100)
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	queues := map[string]QueueInfo{idA: qA}
	require.NotNil(t, s.Pick(0, queues))
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10, 5) // alpha idle; group full

	// beta is heavy, gamma is light: least-service entry picks gamma.
	s.getOrCreateState("beta").AddService(1000)
	idB, qB := makeInfo("beta", now.Add(-10*time.Second))
	idC, qC := makeInfo("gamma", now)
	queues2 := map[string]QueueInfo{idB: qB, idC: qC}

	got := s.Pick(0, queues2)
	require.NotNil(t, got)
	assert.Equal(t, "gamma", got.FlowKey().ID, "least-service entrant wins despite younger head")
}

func TestGroupTurn_EntryOldestWait(t *testing.T) {
	s := newGroupTurn(1, 100)
	s.entryPolicy = groupTurnEntryOldestWait
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	queues := map[string]QueueInfo{idA: qA}
	require.NotNil(t, s.Pick(0, queues))
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10, 5)

	s.getOrCreateState("gamma").AddService(1000) // heavy but oldest head
	idB, qB := makeInfo("beta", now.Add(-1*time.Second))
	idC, qC := makeInfo("gamma", now.Add(-30*time.Second))
	queues2 := map[string]QueueInfo{idB: qB, idC: qC}

	got := s.Pick(0, queues2)
	require.NotNil(t, got)
	assert.Equal(t, "gamma", got.FlowKey().ID, "oldest-wait entrant wins")
}

func TestGroupTurn_CreditResetZeroesService(t *testing.T) {
	s := newGroupTurn(1, 1)
	s.creditPolicy = groupTurnCreditReset
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	require.NotNil(t, s.Pick(0, queues))
	s.getOrCreateState("alpha").AddService(500)
	gtDispatch(s, "alpha") // quota of 1 exhausted

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID)
	assert.Zero(t, s.getOrCreateState("alpha").Service(time.Now(), 0), "reset policy zeroes credit on leave")
}

func TestGroupTurn_CreditKeepRetainsService(t *testing.T) {
	s := newGroupTurn(1, 1)
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}

	require.NotNil(t, s.Pick(0, queues))
	s.getOrCreateState("alpha").AddService(500)
	gtDispatch(s, "alpha")

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID)
	assert.Equal(t, 500.0, s.getOrCreateState("alpha").Service(time.Now(), 0), "keep policy retains credit")
}

func TestGroupTurn_VictimLeastServiceArm(t *testing.T) {
	s := newGroupTurn(2, 100)
	s.victimPolicy = groupTurnVictimLeastService
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-3*time.Second))
	idB, qB := makeInfo("beta", now.Add(-2*time.Second))
	queues := map[string]QueueInfo{idA: qA, idB: qB}
	require.NotNil(t, s.Pick(0, queues)) // both admitted
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 10, 5)
	gtDispatch(s, "beta")
	gtComplete(s, "beta", 10, 5)

	s.getOrCreateState("alpha").AddService(1000) // heavy
	s.getOrCreateState("beta").AddService(10)    // light -> victim under least-service

	idC, qC := makeInfo("gamma", now)
	queues2 := map[string]QueueInfo{idC: qC}
	got := s.Pick(0, queues2)
	require.NotNil(t, got)
	assert.Equal(t, "gamma", got.FlowKey().ID)
	assert.True(t, s.isMember("alpha"), "heavy member keeps its seat")
	assert.False(t, s.isMember("beta"), "least-service member is the victim")
}

func TestGroupTurn_LRS_LeastRecentlyDispatchedFirst(t *testing.T) {
	s := newGroupTurn(4, 100)
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	idB, qB := makeInfo("beta", now)
	queues := map[string]QueueInfo{idA: qA, idB: qB}
	require.NotNil(t, s.Pick(0, queues)) // both admitted

	gtDispatch(s, "alpha") // alpha served recently
	st := s.getOrCreateState("beta")
	st.mu.Lock()
	st.lastDispatch = now.Add(-time.Minute) // beta served long ago
	st.mu.Unlock()

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID, "least recently served member goes first")
}

func TestGroupTurn_OnCompleted_AccumulatesWeightedService(t *testing.T) {
	s := newGroupTurn(4, 4)
	gtDispatch(s, "alpha")
	gtComplete(s, "alpha", 100, 50)
	assert.Equal(t, 200.0, s.getOrCreateState("alpha").Service(time.Now(), 0))
}

func TestGroupTurn_InFlightGuardsHoldExit(t *testing.T) {
	s := newGroupTurn(1, 100)
	s.holdDuration = 50 * time.Millisecond
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	queues := map[string]QueueInfo{idA: qA}
	require.NotNil(t, s.Pick(0, queues))
	gtDispatch(s, "alpha") // in flight, never completes

	st := s.getOrCreateState("alpha")
	st.mu.Lock()
	st.lastActivity = now.Add(-time.Minute) // backdated, but still in flight
	st.mu.Unlock()

	idB, qB := makeInfo("beta", now)
	queues2 := map[string]QueueInfo{idB: qB}
	got := s.Pick(0, queues2)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID, "no evictable member: outsider is gap-filled without membership")
	assert.True(t, s.isMember("alpha"), "in-flight member is neither hold-evicted nor victim-evicted")
	assert.False(t, s.isMember("beta"), "gap-fill does not grant membership")
}

func TestGroupTurn_EvictProgramRemovesMembership(t *testing.T) {
	s := newGroupTurn(4, 4)
	now := time.Now()
	idA, qA := makeInfo("alpha", now)
	queues := map[string]QueueInfo{idA: qA}
	require.NotNil(t, s.Pick(0, queues))
	assert.True(t, s.isMember("alpha"))

	s.EvictProgram("alpha")
	assert.False(t, s.isMember("alpha"))
}

func TestGroupTurn_NoWaitingReturnsNil(t *testing.T) {
	s := newGroupTurn(4, 4)
	queues := map[string]QueueInfo{
		"idle": {Queue: makeQueue("idle", 0, time.Time{}), Metrics: &ProgramMetrics{}, Len: 0},
	}
	assert.Nil(t, s.Pick(0, queues))
}

func TestNewStrategy_GroupTurnWiring(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	require.NoError(t, cfg.validate())
	st, err := newStrategy(cfg)
	require.NoError(t, err)
	gt, ok := st.(*groupTurnStrategy)
	require.True(t, ok)
	assert.Equal(t, 32, gt.groupSize)
	assert.Equal(t, int64(4), gt.turnQuota)
	assert.Equal(t, groupTurnEntryLeastService, gt.entryPolicy)
	assert.Equal(t, groupTurnVictimLongestIdle, gt.victimPolicy)
	assert.Equal(t, groupTurnCreditKeep, gt.creditPolicy)
	assert.Equal(t, 120*time.Second, gt.holdDuration)
	assert.Equal(t, 60.0, gt.halfLifeSeconds)
}

func TestGroupTurn_ConfigValidation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnSize = 0
	assert.Error(t, cfg.validate())

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnQuota = 0
	assert.Error(t, cfg.validate())

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnEntryPolicy = "bogus"
	assert.Error(t, cfg.validate())

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnVictimPolicy = "bogus"
	assert.Error(t, cfg.validate())

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnCreditPolicy = "bogus"
	assert.Error(t, cfg.validate())
}

func TestGroupTurn_WarmthEDF_SavesExpiringWarmMember(t *testing.T) {
	s := newGroupTurn(4, 100)
	s.urgencySeconds = 60 * time.Second
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	idB, qB := makeInfo("beta", now.Add(-1*time.Second))
	queues := map[string]QueueInfo{idA: qA, idB: qB}
	require.NotNil(t, s.Pick(0, queues)) // both admitted

	// alpha dispatched 50s ago: warm, expiring in 10s.
	// beta dispatched 5 minutes ago: cold (loss sunk).
	gtDispatch(s, "alpha")
	stA := s.getOrCreateState("alpha")
	stA.mu.Lock()
	stA.lastDispatch = now.Add(-50 * time.Second)
	stA.mu.Unlock()
	stB := s.getOrCreateState("beta")
	stB.mu.Lock()
	stB.lastDispatch = now.Add(-5 * time.Minute)
	stB.mu.Unlock()

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID, "warm member closest to expiry is saved before the cold one (plain LRS would pick beta)")
}

func TestGroupTurn_WarmthEDF_AllColdFallsBackToLRS(t *testing.T) {
	s := newGroupTurn(4, 100)
	s.urgencySeconds = 60 * time.Second
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	idB, qB := makeInfo("beta", now.Add(-1*time.Second))
	queues := map[string]QueueInfo{idA: qA, idB: qB}
	require.NotNil(t, s.Pick(0, queues))

	// Both cold: LRS order applies (oldest lastDispatch first).
	stA := s.getOrCreateState("alpha")
	stA.mu.Lock()
	stA.lastDispatch = now.Add(-5 * time.Minute)
	stA.mu.Unlock()
	stB := s.getOrCreateState("beta")
	stB.mu.Lock()
	stB.lastDispatch = now.Add(-2 * time.Minute)
	stB.mu.Unlock()

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "alpha", got.FlowKey().ID, "all cold: least recently served (5 min) goes first")
}

func TestGroupTurn_WarmthEDF_NeverDispatchedIsCold(t *testing.T) {
	s := newGroupTurn(4, 100)
	s.urgencySeconds = 60 * time.Second
	now := time.Now()

	idA, qA := makeInfo("alpha", now.Add(-2*time.Second))
	idB, qB := makeInfo("beta", now.Add(-1*time.Second))
	queues := map[string]QueueInfo{idA: qA, idB: qB}
	require.NotNil(t, s.Pick(0, queues))

	// alpha never dispatched (newcomer member, zero lastDispatch); beta warm.
	gtDispatch(s, "beta")
	stB := s.getOrCreateState("beta")
	stB.mu.Lock()
	stB.lastDispatch = now.Add(-30 * time.Second)
	stB.mu.Unlock()

	got := s.Pick(0, queues)
	require.NotNil(t, got)
	assert.Equal(t, "beta", got.FlowKey().ID, "warm member beats the never-dispatched newcomer (no prefix to save)")
}

func newAdaptive(gMax int) *groupTurnStrategy {
	s := newGroupTurn(4, 100)
	s.adaptive = true
	s.gMax = gMax
	s.kneeMax = 120 * time.Second
	s.sb = newSBState(gMax, s.kneeMax)
	return s
}

func TestSB_G_ShrinksSlowlyIntoBandThenGrows(t *testing.T) {
	sb := newSBState(256, 120*time.Second)
	assert.Equal(t, 256, sb.currentG, "G0 = max_num_seqs (non-porous by construction)")

	// 40 minutes at running≈150 (jittered like a real gauge — a perfectly flat
	// reading is a dead sensor and freezes G by design): EWMA converges ~150;
	// shrink is triple-gated (k=3, age >= 15 min, cold admissions) and
	// bounded at -8/update.
	now := time.Now()
	prev := 256
	for i := 1; i <= 40; i++ {
		sb.stepG(int64(150+i%3-1), 256, now.Add(time.Duration(i)*time.Minute), 10)
		g := sb.effectiveG()
		assert.LessOrEqual(t, prev-g, 8, "shrink is non-refill, at most 8 per update")
		assert.GreaterOrEqual(t, g, 16)
		prev = g
	}
	g := sb.effectiveG()
	rBar := sb.runningEWMA
	assert.InDelta(t, 150, rBar, 5)
	assert.GreaterOrEqual(t, float64(g), 1.15*rBar, "never below the band floor")
	assert.LessOrEqual(t, float64(g), 1.45*rBar+8, "reaches the band")

	// Running set jumps to 240: grow is immediate, +32 capped, bounded by gMax.
	for i := 1; i <= 6; i++ {
		sb.stepG(int64(240+i%3-1), 256, now.Add(time.Duration(40+i)*time.Minute), 10)
	}
	assert.Equal(t, 256, sb.effectiveG(), "grows back to the cap")
}

func TestSB_G_ShortLullDoesNotShrink(t *testing.T) {
	sb := newSBState(256, 120*time.Second)
	now := time.Now()
	// Converge with running≈150 (jittered like a real gauge) for 30 min: G
	// settles in the band with the shrink streak reset.
	for i := 1; i <= 30; i++ {
		sb.stepG(int64(150+i%3-1), 256, now.Add(time.Duration(i)*time.Minute), 10)
	}
	gBefore := sb.effectiveG()
	// A single 60s lull poll at r=80: the asymmetric (slow-down) filter keeps
	// rbar high enough that the shrink streak (k=3) cannot complete in one
	// poll — G must not shrink.
	sb.stepG(80, 256, now.Add(31*time.Minute), 10)
	assert.Equal(t, gBefore, sb.effectiveG(), "one lull poll must not shrink G")
	// A second lull poll: streak is 2 of 3 — still no shrink.
	sb.stepG(80, 256, now.Add(32*time.Minute), 10)
	assert.Equal(t, gBefore, sb.effectiveG(), "two lull polls must not shrink G (k=3 guard)")
}

func TestSB_Tau_ColdThenWarmFlipClamped(t *testing.T) {
	sb := newSBState(256, 120*time.Second)
	base := time.Now()

	feed := func(bin0, bin1, bin2 int) {
		// bin ratios: plateau 0.90; cold: 30-60s warm-ish, 60-120s cold.
		for i := 0; i < bin0; i++ {
			sb.feedReEntry(10*time.Second, 90, 100, base.Add(time.Duration(i)*500*time.Millisecond))
		}
		for i := 0; i < bin1; i++ {
			sb.feedReEntry(45*time.Second, 85, 100, base.Add(time.Duration(i)*500*time.Millisecond))
		}
		for i := 0; i < bin2; i++ {
			sb.feedReEntry(90*time.Second, 35, 100, base.Add(time.Duration(i)*500*time.Millisecond))
		}
	}
	// Cold well: 60-120s re-entries miss; bins valid (>=25 samples each);
	// the knee reads the 60s edge (bin 1 upper edge).
	feed(30, 30, 30)
	sb.stepTau(120*time.Second, base.Add(30*time.Second))
	assert.Equal(t, 60*time.Second, sb.knee(), "cold well: knee at the 60s edge")

	// Warm flip over a realistic ~10-minute window (the bin EWMA T_f = 600 s
	// needs the new regime to outvote the old — the sim saw the latch at
	// minute ~20): 60-120s re-entries now hit; the latch jumps the knee to
	// 120s (also the deployment clamp).
	base2 := base.Add(time.Minute)
	for i := 0; i < 60; i++ {
		t := base2.Add(time.Duration(i) * 10 * time.Second)
		sb.feedReEntry(10*time.Second, 90, 100, t)
		sb.feedReEntry(90*time.Second, 85, 100, t)
	}
	sb.stepTau(120*time.Second, base2.Add(11*time.Minute))
	sb.stepTau(120*time.Second, base2.Add(13*time.Minute))
	k := sb.knee()
	assert.Equal(t, 120*time.Second, k, "warm flip latches the knee to 120s (clamped at kneeMax)")
}

func TestSB_Tau_StarvedHolds(t *testing.T) {
	sb := newSBState(256, 120*time.Second)
	base := time.Now()
	for i := 0; i < 40; i++ { // enough to be valid once
		sb.feedReEntry(10*time.Second, 90, 100, base.Add(time.Duration(i)*500*time.Millisecond))
	}
	sb.stepTau(120*time.Second, base.Add(30*time.Second))
	before := sb.knee()
	// 30 minutes of starvation: the decayed sample count falls below the
	// validity gate and the knee must hold.
	late := base.Add(30 * time.Minute)
	sb.stepTau(120*time.Second, late)
	assert.Equal(t, before, sb.knee(), "starved estimator holds the prior knee")
}

func TestSB_AdaptivePickUsesControllerG(t *testing.T) {
	s := newAdaptive(8)
	now := time.Now()
	queues := map[string]QueueInfo{}
	for i := 0; i < 6; i++ {
		id, q := makeInfo(fmt.Sprintf("s%d", i), now.Add(-time.Duration(i)*time.Second))
		queues[id] = q
	}
	got := s.Pick(0, queues)
	require.NotNil(t, got)
	// G0 = 8 > 6 waiting: everyone is a member.
	for i := 0; i < 6; i++ {
		assert.True(t, s.isMember(fmt.Sprintf("s%d", i)))
	}
	// The warmth knee is the controller's bootstrap (60s).
	assert.Equal(t, 60*time.Second, s.warmKnee())
}

func TestSB_ConfigValidation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnAdaptive = true
	cfg.GroupTurnAdaptiveGMax = 8
	assert.Error(t, cfg.validate())

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnAdaptive = true
	cfg.GroupTurnKneeMaxSeconds = 10
	assert.Error(t, cfg.validate())

	cfg = DefaultConfig()
	cfg.Strategy = groupTurnStrategyName
	cfg.GroupTurnAdaptive = true
	require.NoError(t, cfg.validate())
	st, err := newStrategy(cfg)
	require.NoError(t, err)
	gt, ok := st.(*groupTurnStrategy)
	require.True(t, ok)
	assert.True(t, gt.adaptive)
	assert.NotNil(t, gt.sb)
	assert.Equal(t, 256, gt.gMax)
	assert.Equal(t, 120*time.Second, gt.kneeMax)
}

func TestSB_G_DrainFreezeHolds(t *testing.T) {
	sb := newSBState(256, 120*time.Second)
	now := time.Now()
	// Converge with waiting outsiders present (shrink allowed): G enters the band.
	for i := 1; i <= 30; i++ {
		sb.stepG(int64(150+i%3-1), 256, now.Add(time.Duration(i)*time.Minute), 10)
	}
	gBefore := sb.effectiveG()
	// Drain: running falls, nothing waits outside. The drain-freeze must hold G
	// (shrinking without contention is pointless churn).
	for i := 1; i <= 10; i++ {
		sb.stepG(int64(100+i%3-1), 256, now.Add(time.Duration(30+i)*time.Minute), 0)
	}
	assert.Equal(t, gBefore, sb.effectiveG(), "drain (no outsiders waiting) must not shrink G")
}
