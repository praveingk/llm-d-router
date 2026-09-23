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

package weightedlas

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// newTestPolicy builds a policy from a JSON parameters blob, exercising the real
// config path. An empty blob means "all defaults".
func newTestPolicy(t *testing.T, params string) *weightedLAS {
	t.Helper()
	var dec *json.Decoder
	if params != "" {
		dec = json.NewDecoder(bytes.NewReader([]byte(params)))
	}
	p, err := WeightedLASFairnessPolicyFactory("test", dec, nil)
	require.NoError(t, err)
	require.NotNil(t, p)
	policy, ok := p.(*weightedLAS)
	require.True(t, ok, "factory returned %T", p)
	return policy
}

// queuedFlow builds a single-item queue for a flow. It takes no weight: weights are
// operator configuration now, so nothing a request carries can affect its share.
func queuedFlow(id string, priority int, headEnqueue time.Time) *fwkfcmocks.MockFlowQueueAccessor {
	return queuedFlowWithHeaders(id, priority, headEnqueue, map[string]string{})
}

// queuedFlowWithHeaders is the same but lets a test put arbitrary headers on the head
// request, to assert that they are ignored.
func queuedFlowWithHeaders(
	id string, priority int, headEnqueue time.Time, headers map[string]string,
) *fwkfcmocks.MockFlowQueueAccessor {
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: id, Priority: priority},
		PeekV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV: headEnqueue,
			OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{
				FlowKeyV: flowcontrol.FlowKey{ID: id, Priority: priority},
				InferenceRequestV: &fwksched.InferenceRequest{
					FairnessID: id,
					Headers:    headers,
					Objectives: fwksched.RequestObjectives{Priority: priority},
				},
			},
		},
	}
}

// bandOf wraps queues in a priority band accessor that iterates them.
func bandOf(priority int, queues ...flowcontrol.FlowQueueAccessor) *fwkfcmocks.MockPriorityBandAccessor {
	return &fwkfcmocks.MockPriorityBandAccessor{
		PriorityV: priority,
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			for _, q := range queues {
				if !cb(q) {
					return
				}
			}
		},
	}
}

// charge accrues attained service for a flow through the real completion hook.
// promptTokens are weighted 1x, so cost == promptTokens when completion is 0.
func charge(p *weightedLAS, id string, priority int, promptTokens int) {
	req := &fwksched.InferenceRequest{
		FairnessID: id,
		Objectives: fwksched.RequestObjectives{Priority: priority},
	}
	resp := &fwkrc.Response{EndOfStream: true}
	resp.Usage.PromptTokens = promptTokens
	p.ResponseBody(context.Background(), req, resp, nil)
}

// A flow CONFIGURED at weight 3 that has consumed 300 tokens is at 100 tokens per unit
// of weight, so it outranks a weight-1 flow sitting at 150 -- even though its raw
// attained service is twice as high. Unweighted LAS picks beta here.
func TestPick_WeightScalesService(t *testing.T) {
	p := newTestPolicy(t, `{"weightService":1.0,"weightHeadWait":0.0,"halfLifeSeconds":0,
		"flowWeights":{"alpha":3,"beta":1}}`)
	now := time.Now()

	charge(p, "alpha", 0, 300)
	charge(p, "beta", 0, 150)

	band := bandOf(0,
		queuedFlow("alpha", 0, now),
		queuedFlow("beta", 0, now),
	)

	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, got, "expected a queue to be picked")
	assert.Equal(t, "alpha", got.FlowKey().ID)
}

var _ fwkplugin.Plugin = &weightedLAS{}

// A flow that has not been touched for longer than the idle TTL is dropped, so
// neither the state map nor its metric label series grows without bound.
func TestPrune_DropsIdleFlows(t *testing.T) {
	p := newTestPolicy(t, `{"idleTtlSeconds":100,"halfLifeSeconds":0}`)
	now := time.Now()

	charge(p, "stale", 0, 10)
	charge(p, "fresh", 0, 10)
	// Hold "fresh" open by touching it just before the sweep.
	p.stateFor(flowcontrol.FlowKey{ID: "fresh", Priority: 0}).service(now.Add(150*time.Second), 0)

	p.prune(now.Add(150 * time.Second))

	_, staleOK := p.state.Load(flowcontrol.FlowKey{ID: "stale", Priority: 0})
	_, freshOK := p.state.Load(flowcontrol.FlowKey{ID: "fresh", Priority: 0})
	assert.False(t, staleOK, "expected the idle flow to be pruned")
	assert.True(t, freshOK, "expected the recently touched flow to survive")
}

// Two instances of this policy must be able to coexist -- one per priority band is a
// normal configuration -- so their collectors must not collide in one registry.
// Registering package-level collectors with MustRegister is what makes that fatal.
func TestTwoInstances_ShareARegistry(t *testing.T) {
	reg := prometheus.NewRegistry()
	handle := fwkplugin.NewEppHandle(
		context.Background(),
		func() []types.NamespacedName { return nil },
		fwkplugin.WithMetricsRecorder(reg),
	)

	first, err := WeightedLASFairnessPolicyFactory("premium-band", nil, handle)
	require.NoError(t, err)
	require.NotNil(t, first)

	second, err := WeightedLASFairnessPolicyFactory("batch-band", nil, handle)
	require.NoError(t, err, "a second instance must not fail to register its collectors")
	require.NotNil(t, second)
}

// Attained service is published per flow so the achieved share can be compared
// against the configured weights without re-deriving it from throughput.
func TestResponseBody_PublishesAttainedService(t *testing.T) {
	p := newTestPolicy(t, `{"halfLifeSeconds":0}`)

	charge(p, "alpha", 3, 120)

	got := testutil.ToFloat64(p.serviceGauge.WithLabelValues("alpha", "3"))
	assert.InDelta(t, 120.0, got, 1e-9)
}

// A FlowKey is composite, so the same tenant sending at two priorities is two
// separate flows. Their service must not pool: bands dispatch in strict order, so
// charging one band's work against the tenant's queue in another band would
// deprioritize it where it consumed nothing.
//
// Keying state by fairness ID alone would make "tenant" look like the heaviest
// consumer in band 10 on the strength of work it did in band 0, and Pick would choose
// "other" instead.
func TestPick_PerBandIsolation(t *testing.T) {
	p := newTestPolicy(t, `{"weightService":1.0,"weightHeadWait":0.0,"halfLifeSeconds":0}`)
	now := time.Now()

	charge(p, "tenant", 0, 500) // heavy use in a different band
	charge(p, "other", 10, 100) // light use in the band under test

	band := bandOf(10,
		queuedFlow("tenant", 10, now),
		queuedFlow("other", 10, now),
	)

	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, "tenant", got.FlowKey().ID,
		"tenant has consumed nothing in band 10 and should be served there")

	assert.Zero(t, p.stateFor(flowcontrol.FlowKey{ID: "tenant", Priority: 10}).attainedService,
		"service charged at priority 0 must not appear at priority 10")
}

// The actual claim of the feature: over many dispatches, service shares converge on
// the ratio of the configured weights.
func TestPick_ConvergesToWeightRatio(t *testing.T) {
	// Weights are declared by the operator, not by the requests. The empty header argument
	// below is deliberate: it shows the ratio comes from config alone.
	p := newTestPolicy(t, `{"weightService":1.0,"weightHeadWait":0.0,"halfLifeSeconds":0,
		"flowWeights":{"big":4,"mid":2,"small":1}}`)
	now := time.Now()

	band := bandOf(0,
		queuedFlow("big", 0, now),
		queuedFlow("mid", 0, now),
		queuedFlow("small", 0, now),
	)

	const iterations = 3000
	for range iterations {
		got, err := p.Pick(context.Background(), band)
		require.NoError(t, err)
		require.NotNil(t, got)
		charge(p, got.FlowKey().ID, 0, 100)
	}

	service := func(id string) float64 {
		return p.stateFor(flowcontrol.FlowKey{ID: id, Priority: 0}).service(now, 0)
	}
	small := service("small")
	require.Positive(t, small, "every flow should have been served at least once")

	assert.InEpsilon(t, 4.0, service("big")/small, 0.05, "big should receive ~4x the service of small")
	assert.InEpsilon(t, 2.0, service("mid")/small, 0.05, "mid should receive ~2x the service of small")
}

func TestDecay_HalvesAtHalfLife(t *testing.T) {
	p := newTestPolicy(t, `{"halfLifeSeconds":30}`)
	st := p.stateFor(flowcontrol.FlowKey{ID: "alpha", Priority: 0})
	start := time.Now()

	st.addService(100, start, 30)
	assert.InDelta(t, 50.0, st.service(start.Add(30*time.Second), 30), 1e-6)

	fixed := p.stateFor(flowcontrol.FlowKey{ID: "beta", Priority: 0})
	fixed.addService(100, start, 0)
	assert.InDelta(t, 100.0, fixed.service(start.Add(time.Hour), 0), 1e-9, "a zero half-life disables decay")
}

func TestPick_NoCandidates(t *testing.T) {
	p := newTestPolicy(t, "")

	got, err := p.Pick(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, got, "a nil band yields no pick")

	empty := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     0,
		FlowKeyV: flowcontrol.FlowKey{ID: "drained"},
	}
	got, err = p.Pick(context.Background(), bandOf(0, empty))
	require.NoError(t, err)
	assert.Nil(t, got, "an empty queue is not a candidate")
}

func TestConfig_Rejects(t *testing.T) {
	cases := map[string]string{
		"negative service weight": `{"weightService":-1}`,
		"negative half life":      `{"halfLifeSeconds":-1}`,
		"zero sweep interval":     `{"sweepSeconds":0}`,
	}
	for name, params := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := WeightedLASFairnessPolicyFactory("x", json.NewDecoder(bytes.NewReader([]byte(params))), nil)
			require.Error(t, err)
		})
	}
}

// chargeIO accrues service through the real completion hook with BOTH token counts set,
// which is what makes the cost coefficients observable.
func chargeIO(p *weightedLAS, id string, priority, promptTokens, completionTokens int) {
	req := &fwksched.InferenceRequest{
		FairnessID: id,
		Objectives: fwksched.RequestObjectives{Priority: priority},
	}
	resp := &fwkrc.Response{EndOfStream: true}
	resp.Usage.PromptTokens = promptTokens
	resp.Usage.CompletionTokens = completionTokens
	p.ResponseBody(context.Background(), req, resp, nil)
}

// The default cost function must stay 1x input + 2x output, so existing deployments are
// unaffected by the coefficients becoming configurable.
func TestCost_DefaultsAreOneInputTwoOutput(t *testing.T) {
	p := newTestPolicy(t, `{"halfLifeSeconds":0}`)

	chargeIO(p, "alpha", 0, 100, 10)

	assert.InDelta(t, 120.0, testutil.ToFloat64(p.serviceGauge.WithLabelValues("alpha", "0")), 1e-9)
}

// EQUAL OUTPUT THROUGHPUT. On a long-context agentic corpus, input is ~98% of the default
// cost, so the default is effectively prefill-fairness and generation throughput is almost
// unrepresented. Charging output only makes the policy share tok/s instead.
func TestCost_OutputOnlySharesGenerationThroughput(t *testing.T) {
	p := newTestPolicy(t, `{"halfLifeSeconds":0,"costPerInputToken":0,"costPerOutputToken":1}`)

	chargeIO(p, "alpha", 0, 100000, 250)

	assert.InDelta(t, 250.0, testutil.ToFloat64(p.serviceGauge.WithLabelValues("alpha", "0")), 1e-9,
		"a 100k-token prompt must cost nothing when only output is charged")
}

// EQUAL TURNS. A flat per-request cost with both token coefficients at zero makes attained
// service count requests, so equalizing service/weight equalizes request counts -- the unit the
// roundrobin policy uses, reached here through service accounting rather than by cycling turns.
func TestCost_PerRequestOnlyCountsTurns(t *testing.T) {
	p := newTestPolicy(t, `{"halfLifeSeconds":0,"costPerRequest":1,"costPerInputToken":0,"costPerOutputToken":0}`)

	chargeIO(p, "alpha", 0, 100000, 5000)
	chargeIO(p, "alpha", 0, 7, 1)

	assert.InDelta(t, 2.0, testutil.ToFloat64(p.serviceGauge.WithLabelValues("alpha", "0")), 1e-9,
		"two requests of wildly different size must cost 2 when only requests are charged")
}

// A blend is the point of having three coefficients: a fixed per-request overhead plus token
// costs is closer to real GPU cost than either extreme.
func TestCost_BlendsAllThreeCoefficients(t *testing.T) {
	p := newTestPolicy(t, `{"halfLifeSeconds":0,"costPerRequest":50,"costPerInputToken":0.5,"costPerOutputToken":3}`)

	chargeIO(p, "alpha", 0, 200, 10)

	// 50 + 0.5*200 + 3*10 = 180
	assert.InDelta(t, 180.0, testutil.ToFloat64(p.serviceGauge.WithLabelValues("alpha", "0")), 1e-9)
}

// All-zero coefficients would make every cost 0, so nothing is ever charged, every flow sits
// at zero service forever and the policy silently degenerates into head-wait ordering. That
// must be a startup error rather than a plausible-looking null result.
func TestConfig_RejectsAllZeroCostCoefficients(t *testing.T) {
	_, err := WeightedLASFairnessPolicyFactory("x",
		json.NewDecoder(bytes.NewReader([]byte(`{"costPerRequest":0,"costPerInputToken":0,"costPerOutputToken":0}`))), nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "cost")
}

func TestConfig_RejectsNegativeCostCoefficient(t *testing.T) {
	_, err := WeightedLASFairnessPolicyFactory("x",
		json.NewDecoder(bytes.NewReader([]byte(`{"costPerOutputToken":-1}`))), nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "costPerOutputToken")
}

func TestConfig_FlowWeightsFromOperatorConfig(t *testing.T) {
	t.Parallel()
	p := newTestPolicy(t, `{"flowWeights":{"team-a":7,"team-b":3},"defaultFlowWeight":1}`)

	assert.Equal(t, 7.0, p.cfg.FlowWeights["team-a"])
	assert.Equal(t, 3.0, p.cfg.FlowWeights["team-b"])
	assert.Equal(t, 1.0, p.cfg.DefaultFlowWeight)
}

// An unset defaultFlowWeight must be 1, not 0: a zero default would divide by zero for
// every unlisted tenant.
func TestConfig_DefaultFlowWeightDefaultsToOne(t *testing.T) {
	t.Parallel()
	p := newTestPolicy(t, `{"flowWeights":{"team-a":7}}`)

	assert.Equal(t, 1.0, p.cfg.DefaultFlowWeight)
}

// Weights are operator input now, so a bad value is a deployment error and must be loud at
// startup rather than clamped into something plausible.
func TestConfig_RejectsNonPositiveDeclaredWeight(t *testing.T) {
	t.Parallel()
	for _, blob := range []string{
		`{"flowWeights":{"team-a":0}}`,
		`{"flowWeights":{"team-a":-3}}`,
		`{"defaultFlowWeight":0}`,
	} {
		_, err := WeightedLASFairnessPolicyFactory("x",
			json.NewDecoder(bytes.NewReader([]byte(blob))), nil)
		require.Error(t, err, "blob %s should be rejected", blob)
		assert.Contains(t, err.Error(), "weight")
	}
}

func TestWeightFor_ResolvesFromConfigNotHeader(t *testing.T) {
	t.Parallel()
	p := newTestPolicy(t, `{"flowWeights":{"team-a":7,"team-b":3},"defaultFlowWeight":2}`)

	assert.Equal(t, 7.0, p.weightFor(flowcontrol.FlowKey{ID: "team-a", Priority: 0}))
	assert.Equal(t, 3.0, p.weightFor(flowcontrol.FlowKey{ID: "team-b", Priority: 0}))
	assert.Equal(t, 2.0, p.weightFor(flowcontrol.FlowKey{ID: "unlisted", Priority: 0}),
		"an unlisted tenant takes the default")
}

// Weight is a property of the tenant, not of the band, so the same ID at two priorities
// resolves to the same weight even though the two remain separate flows for accounting.
func TestWeightFor_KeysOnIDNotPriority(t *testing.T) {
	t.Parallel()
	p := newTestPolicy(t, `{"flowWeights":{"team-a":7}}`)

	assert.Equal(t, 7.0, p.weightFor(flowcontrol.FlowKey{ID: "team-a", Priority: 0}))
	assert.Equal(t, 7.0, p.weightFor(flowcontrol.FlowKey{ID: "team-a", Priority: -10}))
}

// A caller cannot influence its own share. The router defines no weight header at all now, so
// this sends a raw, unrecognised one -- the shape a client would forge -- and asserts the
// policy still uses the configured weight. This is the North Star non-goal: a weight sets the
// relative price of capacity, and a request may not name its own price.
func TestWeightFor_IgnoresAnyWeightLikeHeader(t *testing.T) {
	t.Parallel()
	p := newTestPolicy(t, `{"flowWeights":{"team-a":7}}`)
	band := bandOf(0, queuedFlowWithHeaders("team-a", 0, time.Now(), map[string]string{
		"x-llm-d-inference-fairness-weight": "999",
	}))

	got, err := p.Pick(context.Background(), band)

	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, 7.0, testutil.ToFloat64(p.weightGauge.WithLabelValues("team-a", "0")),
		"the gauge must report the CONFIGURED weight, never the header's 999")
}
