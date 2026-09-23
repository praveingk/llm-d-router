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
// Package weightedlas implements a flow-control fairness policy: least attained
// service, scaled by a per-flow share weight.
package weightedlas

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// WeightedLASFairnessPolicyType is the registration type for this policy.
const WeightedLASFairnessPolicyType = "weighted-las-fairness-policy"

const (
	// defaultFlowWeight applies to a flow that never declares one. Its absolute value
	// is unobservable; only ratios between flows affect scheduling.
	defaultFlowWeight = 1.0
)

// Config is the JSON parameters block for the policy.
type Config struct {
	WeightService   float64 `json:"weightService,omitempty"`
	WeightHeadWait  float64 `json:"weightHeadWait,omitempty"`
	HalfLifeSeconds float64 `json:"halfLifeSeconds,omitempty"`
	IdleTTLSeconds  float64 `json:"idleTtlSeconds,omitempty"`
	SweepSeconds    float64 `json:"sweepSeconds,omitempty"`

	// FlowWeights is the operator's weight table, keyed by fairness ID. The North Star
	// refuses numeric weights on the wire -- "a weight sets the relative price of capacity,
	// and a request may not name its own price" -- so entitlement is declared here and a
	// request carries only the identity it is resolved against.
	//
	// This map is the interim carrier until the entitlement object ships. Nothing outside
	// weightFor depends on where a weight came from, so that swap stays local.
	//
	// There is deliberately no ceiling on a declared weight. The old maxFlowWeight existed
	// only to cap a client-supplied value; operator config wants a loud startup error on a
	// typo, not a silent clamp that hides it.
	FlowWeights map[string]float64 `json:"flowWeights,omitempty"`

	// DefaultFlowWeight applies to any fairness ID absent from FlowWeights, including the
	// default tenant that unstamped callers fall into. Only ratios between flows affect
	// scheduling, so its absolute value is unobservable.
	DefaultFlowWeight float64 `json:"defaultFlowWeight,omitempty"`

	// The cost function. Attained service is charged per completed request as
	//
	//	cost = CostPerRequest + CostPerInputToken*promptTokens + CostPerOutputToken*completionTokens
	//
	// and it is this quantity whose share converges on the ratio of the flows' weights. The
	// coefficients therefore choose WHAT is being shared, which matters more than it sounds:
	// on a long-context agentic corpus (ISL ~130k, OSL ~1k) input is ~98% of the default cost,
	// so the default shares PREFILL and generation throughput is almost unrepresented.
	//
	//	0 / 1 / 2    default -- roughly "GPU work", output weighted 2x as on program-aware LAS
	//	0 / 0 / 1    equal output tokens per second
	//	0 / 1 / 0    equal prefill
	//	1 / 0 / 0    equal request counts -- the unit the roundrobin policy uses, reached here
	//	             through decayed service accounting rather than by cycling turns
	//	50 / 0.5 / 3 a blend: fixed per-request overhead plus token costs
	//
	// All three must be >= 0 and not all zero. All-zero would make every cost 0, so nothing is
	// ever charged, every flow sits at zero service forever, and the policy silently degenerates
	// into head-wait-only ordering while still looking configured.
	CostPerRequest     float64 `json:"costPerRequest,omitempty"`
	CostPerInputToken  float64 `json:"costPerInputToken,omitempty"`
	CostPerOutputToken float64 `json:"costPerOutputToken,omitempty"`
}

// DefaultConfig returns the configuration used when no parameters are supplied.
func DefaultConfig() Config {
	return Config{
		WeightService:   0.8,
		WeightHeadWait:  0.2,
		HalfLifeSeconds: 60,
		IdleTTLSeconds:  3600,
		SweepSeconds:    300,
		// Unlisted tenants share equally. Must not be 0: weightFor divides by it.
		DefaultFlowWeight: 1,
		// 0/1/2 reproduces the previously hardcoded cost function byte for byte.
		CostPerRequest:     0,
		CostPerInputToken:  1,
		CostPerOutputToken: 2,
	}
}

func (c Config) validate() error {
	switch {
	case c.WeightService < 0:
		return fmt.Errorf("weightService must be >= 0, got %v", c.WeightService)
	case c.WeightHeadWait < 0:
		return fmt.Errorf("weightHeadWait must be >= 0, got %v", c.WeightHeadWait)
	case c.HalfLifeSeconds < 0:
		return fmt.Errorf("halfLifeSeconds must be >= 0, got %v", c.HalfLifeSeconds)
	case c.IdleTTLSeconds < 0:
		return fmt.Errorf("idleTtlSeconds must be >= 0, got %v", c.IdleTTLSeconds)
	case c.SweepSeconds <= 0:
		return fmt.Errorf("sweepSeconds must be > 0, got %v", c.SweepSeconds)
	case c.CostPerRequest < 0:
		return fmt.Errorf("costPerRequest must be >= 0, got %v", c.CostPerRequest)
	case c.CostPerInputToken < 0:
		return fmt.Errorf("costPerInputToken must be >= 0, got %v", c.CostPerInputToken)
	case c.CostPerOutputToken < 0:
		return fmt.Errorf("costPerOutputToken must be >= 0, got %v", c.CostPerOutputToken)
	case c.CostPerRequest == 0 && c.CostPerInputToken == 0 && c.CostPerOutputToken == 0:
		return fmt.Errorf("cost coefficients cannot all be zero: nothing would ever be charged, " +
			"so every flow would sit at zero attained service and the policy would silently " +
			"degenerate into head-wait-only ordering")
	case c.DefaultFlowWeight <= 0:
		return fmt.Errorf("defaultFlowWeight must be > 0, got %v", c.DefaultFlowWeight)
	}
	// Declared weights are operator input, so a bad one is a deployment error: fail loudly at
	// startup rather than clamping it into something that looks deliberate.
	for id, w := range c.FlowWeights {
		if w <= 0 {
			return fmt.Errorf("flowWeights[%q]: weight must be > 0, got %v", id, w)
		}
	}
	return nil
}

var (
	_ flowcontrol.FairnessPolicy  = &weightedLAS{}
	_ fwkrc.ResponseBodyProcessor = &weightedLAS{}
)

type weightedLAS struct {
	name string
	cfg  Config

	// Keyed by FlowKey, so one tenant sending at two priorities stays two flows with
	// separate service -- bands dispatch in strict order, and pooling them would let
	// work done in one band deprioritize the tenant in another.
	state sync.Map // flowcontrol.FlowKey -> *flowState

	// Unix-nano instant of the last idle sweep. Pruning rides Pick on a duty cycle, so
	// the policy owns no goroutine.
	lastSweep atomic.Int64

	// Per-instance, labelled with the instance name: two instances of this policy must
	// be able to share one registry, which package-level collectors cannot.
	serviceGauge *prometheus.GaugeVec
	weightGauge  *prometheus.GaugeVec
}

// WeightedLASFairnessPolicyFactory constructs the policy from its parameters block.
func WeightedLASFairnessPolicyFactory(
	name string,
	parameters *json.Decoder,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	cfg := DefaultConfig()
	if parameters != nil {
		if err := parameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", WeightedLASFairnessPolicyType, name, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", WeightedLASFairnessPolicyType, name, err)
	}
	if name == "" {
		name = WeightedLASFairnessPolicyType
	}

	attained, weight := newCollectors(name)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			for _, c := range []prometheus.Collector{attained, weight} {
				if err := reg.Register(c); err != nil {
					return nil, fmt.Errorf("%s plugin %q: registering metrics: %w",
						WeightedLASFairnessPolicyType, name, err)
				}
			}
		}
	}
	return &weightedLAS{name: name, cfg: cfg, serviceGauge: attained, weightGauge: weight}, nil
}

func (p *weightedLAS) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: WeightedLASFairnessPolicyType, Name: p.name}
}

// NewState returns nil: state is keyed by flow, not by band.
func (p *weightedLAS) NewState(_ context.Context) any { return nil }

// Pick serves the flow with the lowest attained service per unit of weight, nudged by
// how long its head request has waited.
//
// Dividing service by weight is the whole mechanism: equalizing service/weight drives
// the flows' service shares toward the ratio of their weights. rangeNormalize is
// monotone, so the service term alone ranks flows by ascending service/weight; the
// head-wait term overrides it when the normalized spread is small, which is what keeps
// a starved newcomer moving.
func (p *weightedLAS) Pick(
	_ context.Context,
	band flowcontrol.PriorityBandAccessor,
) (flowcontrol.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}

	type entry struct {
		queue flowcontrol.FlowQueueAccessor
		// service is attained service DIVIDED BY the flow's share weight. Only this ratio
		// is equalized; raw attained service ends up proportional to weight, not equal.
		service    float64
		headWaitMs float64
	}

	now := time.Now()
	entries := make([]entry, 0, 8)
	// Seeded to infinity so the first entry sets both bounds without a special case. With
	// no entries they stay unused, since the scoring loop below does not run.
	minService, maxService := math.Inf(1), math.Inf(-1)
	minWait, maxWait := math.Inf(1), math.Inf(-1)

	// Only active queues are visited, which is enough: decay is time-based, so an idle
	// flow's service ages out without being seen here.
	band.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}

		key := queue.FlowKey()
		st := p.stateFor(key)
		flowWeight := p.weightFor(key)
		p.weightGauge.WithLabelValues(key.ID, strconv.Itoa(key.Priority)).Set(flowWeight)

		// The one difference from plain LAS: score on service per unit of share weight.
		service := st.service(now, p.cfg.HalfLifeSeconds) / flowWeight
		headWaitMs := max(float64(now.Sub(head.EnqueueTime()).Milliseconds()), 0)

		entries = append(entries, entry{queue, service, headWaitMs})
		minService, maxService = min(minService, service), max(maxService, service)
		minWait, maxWait = min(minWait, headWaitMs), max(maxWait, headWaitMs)
		return true
	})

	p.maybePrune(now)

	var best flowcontrol.FlowQueueAccessor
	bestScore, bestWait := math.Inf(-1), math.Inf(-1)
	for _, e := range entries {
		// Invert service: lower attained service per unit of weight -> higher score.
		normService := 1 - rangeNormalize(e.service, minService, maxService)
		normWait := rangeNormalize(e.headWaitMs, minWait, maxWait)
		score := p.cfg.WeightService*normService + p.cfg.WeightHeadWait*normWait

		// Tie-break on the longer head wait, so equal scores dispatch in arrival order
		// rather than in map iteration order.
		if score > bestScore || (score == bestScore && e.headWaitMs > bestWait) {
			bestScore, bestWait, best = score, e.headWaitMs, e.queue
		}
	}
	return best, nil
}

// ResponseBody charges the flow for the tokens its request consumed. Only the final
// chunk carries a usage block.
func (p *weightedLAS) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	response *fwkrc.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	cost := p.cfg.CostPerRequest +
		p.cfg.CostPerInputToken*float64(response.Usage.PromptTokens) +
		p.cfg.CostPerOutputToken*float64(response.Usage.CompletionTokens)
	if cost == 0 {
		return
	}
	key := flowcontrol.FlowKey{ID: fairnessIDFor(request), Priority: request.Objectives.Priority}
	service := p.stateFor(key).addService(cost, time.Now(), p.cfg.HalfLifeSeconds)
	p.serviceGauge.WithLabelValues(key.ID, strconv.Itoa(key.Priority)).Set(service)
}

// flowState is one flow's accounting. Service decays lazily rather than on a timer:
// serviceAsOf is the instant attainedService is accurate for, and every read or write
// first applies the decay owed since then. So an idle flow ages out without anyone
// visiting it, and serviceAsOf doubles as its last-touched time.
type flowState struct {
	mu              sync.Mutex
	attainedService float64
	serviceAsOf     time.Time
}

// addService folds in the decay owed since serviceAsOf, charges cost, and returns the
// new total. Decay is exponential: with a 60s half-life, a flow at 100 that goes quiet
// for 120s comes back at 25. A half-life of 0 disables decay, and a backwards clock is
// ignored rather than decayed negatively.
func (s *flowState) addService(cost float64, now time.Time, halfLife float64) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.serviceAsOf.IsZero() {
		s.serviceAsOf = now
	} else if elapsed := now.Sub(s.serviceAsOf).Seconds(); elapsed > 0 {
		if halfLife > 0 {
			s.attainedService *= math.Exp2(-elapsed / halfLife)
		}
		s.serviceAsOf = now
	}

	s.attainedService += cost
	return s.attainedService
}

// service is the read: it folds in decay owed and returns the total, charging nothing.
func (s *flowState) service(now time.Time, halfLife float64) float64 {
	return s.addService(0, now, halfLife)
}

func (s *flowState) lastTouch() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.serviceAsOf
}

func (p *weightedLAS) stateFor(key flowcontrol.FlowKey) *flowState {
	if v, ok := p.state.Load(key); ok {
		if st, ok := v.(*flowState); ok {
			return st
		}
	}
	actual, _ := p.state.LoadOrStore(key, &flowState{})
	st, ok := actual.(*flowState)
	if !ok {
		return &flowState{}
	}
	return st
}

// maybePrune runs an idle sweep at most once per sweepSeconds.
// weightFor resolves a flow's weight from the operator's table, falling back to
// DefaultFlowWeight for any ID the table does not name -- including the default tenant that
// unstamped callers fall into.
//
// Keyed on ID alone: weight is a property of the tenant, while FlowKey is composite so that
// the same tenant at two priorities keeps separate service accounting.
//
// There is deliberately no per-flow weight memory and no header read. The router defines no
// weight header at all, so nothing a caller sends can change its share -- which is the point:
// a request may not name its own price.
func (p *weightedLAS) weightFor(key flowcontrol.FlowKey) float64 {
	if w, ok := p.cfg.FlowWeights[key.ID]; ok && w > 0 {
		return w
	}
	return p.cfg.DefaultFlowWeight
}

func (p *weightedLAS) maybePrune(now time.Time) {
	if p.cfg.IdleTTLSeconds <= 0 {
		return
	}
	interval := int64(p.cfg.SweepSeconds * float64(time.Second))
	last := p.lastSweep.Load()
	if last != 0 && now.UnixNano()-last < interval {
		return
	}
	if !p.lastSweep.CompareAndSwap(last, now.UnixNano()) {
		return // another caller owns this interval
	}
	p.prune(now)
}

// prune drops flows untouched for longer than the idle TTL, bounding the state map and
// the per-flow metric series.
func (p *weightedLAS) prune(now time.Time) {
	ttl := time.Duration(p.cfg.IdleTTLSeconds * float64(time.Second))
	p.state.Range(func(key, value any) bool {
		st, ok := value.(*flowState)
		if !ok {
			p.state.Delete(key)
			return true
		}
		// A zero timestamp is a freshly created entry Pick has not read yet.
		if touched := st.lastTouch(); touched.IsZero() || now.Sub(touched) <= ttl {
			return true
		}
		p.state.Delete(key)
		if fk, ok := key.(flowcontrol.FlowKey); ok {
			priority := strconv.Itoa(fk.Priority)
			p.serviceGauge.DeleteLabelValues(fk.ID, priority)
			p.weightGauge.DeleteLabelValues(fk.ID, priority)
		}
		return true
	})
}

// rangeNormalize maps v into [0,1] across the observed range. No spread yields 0.5, a
// neutral term.
func rangeNormalize(v, minV, maxV float64) float64 {
	if maxV == minV {
		return 0.5
	}
	return (v - minV) / (maxV - minV)
}

func fairnessIDFor(req *fwksched.InferenceRequest) string {
	if req == nil || req.FairnessID == "" {
		return metadata.DefaultFairnessID
	}
	return req.FairnessID
}

func newCollectors(name string) (attained, weight *prometheus.GaugeVec) {
	labels := prometheus.Labels{"policy": name}
	attained = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:        "weighted_las_attained_service_tokens",
		Help:        metricsutil.HelpMsgWithStability("Time-decayed attained service in weighted tokens per fairness flow.", compbasemetrics.ALPHA),
		ConstLabels: labels,
	}, []string{"flow_id", "priority"})
	weight = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:        "weighted_las_flow_weight",
		Help:        metricsutil.HelpMsgWithStability("Share weight in force for a fairness flow, as declared by request header.", compbasemetrics.ALPHA),
		ConstLabels: labels,
	}, []string{"flow_id", "priority"})
	return attained, weight
}
