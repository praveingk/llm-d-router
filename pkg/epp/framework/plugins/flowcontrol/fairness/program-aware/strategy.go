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
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Strategy is the fairness scheduling policy. All methods must be safe for
// concurrent use.
type Strategy interface {
	Name() string
	Pick(bandPriority int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor
	OnPreRequest(metrics *ProgramMetrics, request *fwksched.InferenceRequest)
	OnCompleted(metrics *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response)
	EvictProgram(id string)
	Collectors() []prometheus.Collector
}

type QueueInfo struct {
	Queue   flowcontrol.FlowQueueAccessor
	Metrics *ProgramMetrics
	Len     int
}

func newStrategy(cfg Config) (Strategy, error) {
	switch cfg.Strategy {
	case "", "las":
		return &LASStrategy{
			weightService:   cfg.LASWeightService,
			weightHeadWait:  cfg.LASWeightHeadWait,
			halfLifeSeconds: cfg.LASHalfLifeSeconds,
		}, nil
	case turnPriorityStrategyName:
		return &turnPriorityStrategy{
			timeWeight:        cfg.TurnPriorityTimeWeight,
			inactivitySeconds: cfg.TurnPriorityInactivitySeconds,
		}, nil
	case groupTurnStrategyName:
		gt := &groupTurnStrategy{
			groupSize:              cfg.GroupTurnSize,
			turnQuota:              cfg.GroupTurnQuota,
			entryPolicy:            cfg.GroupTurnEntryPolicy,
			victimPolicy:           cfg.GroupTurnVictimPolicy,
			creditPolicy:           cfg.GroupTurnCreditPolicy,
			holdDuration:           time.Duration(cfg.GroupTurnHoldSeconds * float64(time.Second)),
			halfLifeSeconds:        cfg.GroupTurnHalfLifeSeconds,
			urgencySeconds:         time.Duration(cfg.GroupTurnUrgencySeconds * float64(time.Second)),
			adaptive:               cfg.GroupTurnAdaptive,
			gMax:                   cfg.GroupTurnAdaptiveGMax,
			kneeMax:                time.Duration(cfg.GroupTurnKneeMaxSeconds * float64(time.Second)),
			coldBudgetTokensPerSec: cfg.GroupTurnColdBudgetTokensPerSec,
			coldBudgetTenants:      cfg.GroupTurnColdBudgetTenants,
		}
		// The knee-keyed estimator is the production charge source (see
		// strategy_groupturn_cold.go); tests and future sources override it.
		gt.coldCharge = gt.kneeKeyedColdCharge
		if gt.adaptive {
			gt.sb = newSBState(gt.gMax, gt.kneeMax)
		}
		return gt, nil
	default:
		return nil, fmt.Errorf("unknown scoring strategy %q: only \"las\", \"turn-priority\" and \"group-turn\" are supported", cfg.Strategy)
	}
}

func rangeNormalize(v, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	return (v - min) / (max - min)
}
