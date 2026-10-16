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
