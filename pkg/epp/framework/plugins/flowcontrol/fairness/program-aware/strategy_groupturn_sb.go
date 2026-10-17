package programaware

// Self-balancing layer for the group-turn strategy: two deadband trackers
// replacing the load-dependent constants with runtime control, per the
// design in task/turn-group-scheduler/logs/p6b-selfbalance-theory.md and
// the offline results in logs/p6-selfbalance-sim.md.
//
//	G controller: group size tracks the engine's running set (asymmetric
//	  EWMA, band [0.15, 0.45]·r̄, grow immediately in +32 jumps, shrink
//	  slowly via non-refill only, triple-gated; G₀ = max_num_seqs is
//	  non-porous by construction). Provable: cannot oscillate through the
//	  plant (deadband + near-zero plateau gain + one-sided rates +
//	  exogenous target).
//	τ controller: the warmth-EDF knee tracks the measured re-entry hit
//	  curve (log-gap binned token-ratio EWMAs, relative threshold
//	  plateau−0.20, geometric tracker with deadband + flip latch). CLAMPED
//	  at kneeMaxSeconds: the free-ratchet gains hit by breaking the
//	  fairness envelope (sim finding) — the clamp is a deployment bound.
//
// Fairness is never adaptive: entry, credit, hold and quota stay fixed.

import (
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Controller observability: the live group-size target and warmth knee.
var (
	groupTurnGroupTarget = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_group_target",
			Help:      metricsutil.HelpMsgWithStability("Self-balancing G controller's current group-size target.", compbasemetrics.ALPHA),
		})

	groupTurnKneeSeconds = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "program_aware_group_turn_knee_seconds",
			Help:      metricsutil.HelpMsgWithStability("Self-balancing τ controller's current warmth-EDF knee (seconds).", compbasemetrics.ALPHA),
		})
)

const (
	sbGCadence      = 60 * time.Second   // G controller sampling cadence
	sbGEWMAUp       = 120 * time.Second  // fast toward danger (rising running set)
	sbGEWMADown     = 600 * time.Second  // slow away (lull dips must not shrink)
	sbGGrowStep     = 32                 // max seats added per update
	sbGShrinkStep   = 8                  // max seats dropped per update (non-refill)
	sbGSensorFreeze = 3                  // flat-lined polls before freezing G
	sbShrinkMinAge  = 15 * time.Minute   // no shrink during cold start
	sbShrinkMaxAdm  = 8.0                // no shrink while admissions hot (per min)
	sbTauEvalEvery  = 120 * time.Second  // τ readout cadence
	sbTauEWMA       = 600 * time.Second  // bin estimator time constant
	sbTauDeadband   = 0.35               // octaves
	sbTauFlipSignal = 0.75               // octaves; latch trigger
	sbTauLatchLock  = 20 * time.Minute   // flip direction lock
	sbTauStep       = 1.10               // geometric step toward τ̂
	sbTauCrossFreez = 10 * time.Minute   // τ frozen after any G action
	sbTauMinSamples = 30.0               // re-entry samples / 10 min to act
	sbBinMinSamples = 25.0               // effective samples for a valid bin
	sbKneeFloor     = 30 * time.Second
)

// sbTokenEWMA is a token-ratio EWMA over one gap bin: exponentially decayed
// sums of cached and prompt tokens plus a decayed sample count.
type sbTokenEWMA struct {
	cached float64
	prompt float64
	n      float64
}

func (b *sbTokenEWMA) add(cached, prompt float64, now time.Time, lastT time.Time, tc time.Duration) {
	if b.n > 0 && tc > 0 {
		decay := math.Exp(-now.Sub(lastT).Seconds() / tc.Seconds())
		b.cached *= decay
		b.prompt *= decay
		b.n *= decay
	}
	b.cached += cached
	b.prompt += prompt
	b.n++
}

func (b *sbTokenEWMA) ratio() float64 {
	if b.prompt <= 0 {
		return 0
	}
	return b.cached / b.prompt
}

// sbState is the controller state, guarded by mu.
type sbState struct {
	mu sync.Mutex

	// G controller
	runningEWMA   float64
	lastSampleT   time.Time
	lastR         float64
	flatPolls     int
	currentG      int
	shrinkStreak  int     // consecutive polls above the shrink band
	admitRate     float64 // entrant admissions per minute (5-min EWMA)
	admitWindow   float64 // admissions seen since last cadence tick
	admitLastT    time.Time
	startedAt     time.Time
	lastGActionT  time.Time
	gLastTick     time.Time
	tauLastTick   time.Time

	// τ controller (gap bins: [0,30) plateau-ref; 30-60, 60-120, 120-240,
	// 240-480; [480,inf) floor-ref)
	bins         [6]sbTokenEWMA
	binsLastT    [6]time.Time
	sampleCount  float64 // decayed re-entry samples (~10 min memory)
	sampleLastT  time.Time
	currentKnee  time.Duration
	evalStreak   int // consecutive evaluations outside deadband
	flipStreak   int // consecutive evaluations past the flip signal
	latchDir     int // -1 down, +1 up, 0 none
	latchUntil   time.Time
}

func newSBState(gMax int, kneeMax time.Duration) *sbState {
	now := time.Now()
	return &sbState{
		currentG:     gMax,
		startedAt:    now,
		admitLastT:   now,
		sampleLastT:  now,
		gLastTick:    now,
		tauLastTick:  now,
		currentKnee:  60 * time.Second,
	}
}

// tickG reports whether a G-controller cadence slot is due and claims it.
func (sb *sbState) tickG(now time.Time) bool {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	if now.Sub(sb.gLastTick) < sbGCadence {
		return false
	}
	sb.gLastTick = now
	return true
}

// tickTau reports whether a τ-controller evaluation slot is due and claims it.
func (sb *sbState) tickTau(now time.Time) bool {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	if now.Sub(sb.tauLastTick) < sbTauEvalEvery {
		return false
	}
	sb.tauLastTick = now
	return true
}

// kneeIndex maps a dispatch gap to a bin index.
func sbKneeBin(gap time.Duration) int {
	switch s := gap.Seconds(); {
	case s < 30:
		return 0
	case s < 60:
		return 1
	case s < 120:
		return 2
	case s < 240:
		return 3
	case s < 480:
		return 4
	default:
		return 5
	}
}

// feedReEntry records one completed re-entry outcome into the τ estimator.
// gap is the session's inter-dispatch gap captured at dispatch time.
func (sb *sbState) feedReEntry(gap time.Duration, cached, prompt float64, now time.Time) {
	if prompt <= 0 {
		return
	}
	sb.mu.Lock()
	defer sb.mu.Unlock()
	i := sbKneeBin(gap)
	sb.bins[i].add(cached, prompt, now, sb.binsLastT[i], sbTauEWMA)
	sb.binsLastT[i] = now
	// decay the global sample count toward a ~10-minute memory
	elapsed := now.Sub(sb.sampleLastT).Seconds()
	if elapsed > 0 {
		sb.sampleCount *= math.Exp(-elapsed / 600)
	}
	sb.sampleCount++
	sb.sampleLastT = now
}

// noteAdmission counts an entrant admission for the shrink gate's rate EWMA.
func (sb *sbState) noteAdmission() {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	sb.admitWindow++
}

// runningCount is the router-outstanding request count (dispatched −
// completed), the always-available proxy for the engine's running set.
func (s *groupTurnStrategy) sbRunningCount() int64 {
	var total int64
	s.state.Range(func(_, v any) bool {
		if st, ok := v.(*groupTurnState); ok {
			_, _, inFlight, _, _ := st.snapshot()
			total += inFlight
		}
		return true
	})
	return total
}

// stepG runs one G-controller update (call at most once per sbGCadence).
// waitingOutside is the current count of queued non-members (drain-freeze:
// with nothing waiting outside, membership gates nothing and shrinking is
// pointless churn — hold).
func (sb *sbState) stepG(r int64, gMax int, now time.Time, waitingOutside int) {
	sb.mu.Lock()
	defer sb.mu.Unlock()

	// Sensor interlock: a flat-lined reading freezes G.
	if sb.lastSampleT.IsZero() {
		sb.runningEWMA = float64(r)
	} else {
		elapsed := now.Sub(sb.lastSampleT).Seconds()
		if elapsed > 0 {
			if math.Abs(float64(r)-sb.runningEWMA) < 0.5 && float64(r) == sb.lastR {
				sb.flatPolls++
				if sb.flatPolls >= sbGSensorFreeze {
					sb.lastSampleT = now
					sb.lastR = float64(r)
					return // frozen
				}
			} else {
				sb.flatPolls = 0
			}
			tc := sbGEWMAUp.Seconds()
			if float64(r) < sb.runningEWMA {
				tc = sbGEWMADown.Seconds()
			}
			alpha := 1 - math.Exp(-elapsed/tc)
			sb.runningEWMA += alpha * (float64(r) - sb.runningEWMA)
		}
	}
	sb.lastSampleT = now
	sb.lastR = float64(r)

	rBar := sb.runningEWMA
	// Admission-rate EWMA (5-minute) for the shrink gate.
	minutes := now.Sub(sb.admitLastT).Minutes()
	if minutes > 0 {
		inst := sb.admitWindow / minutes
		alpha := 1 - math.Exp(-minutes/5)
		sb.admitRate += alpha * (inst - sb.admitRate)
		sb.admitWindow = 0
		sb.admitLastT = now
	}

	mLo := math.Max(16, 0.15*rBar)
	mHi := math.Max(48, 0.45*rBar)
	rest := rBar + 0.30*rBar

	switch {
	case float64(sb.currentG)-rBar < mLo:
		// Below the band: grow immediately, capped at the deployment bound.
		target := math.Min(rest, math.Min(float64(sb.currentG+sbGGrowStep), float64(gMax)))
		if target > float64(sb.currentG) {
			sb.currentG = int(target)
			sb.lastGActionT = now
			sb.shrinkStreak = 0
		}
	case float64(sb.currentG)-rBar > mHi:
		// Drain-freeze: no outsiders waiting means membership gates nothing;
		// shrinking would only churn the tail.
		if waitingOutside == 0 {
			sb.shrinkStreak = 0
			break
		}
		sb.shrinkStreak++
		// Triple gate: past cold start, admissions not hot, persistence.
		if sb.shrinkStreak >= 3 &&
			now.Sub(sb.startedAt) >= sbShrinkMinAge &&
			sb.admitRate < sbShrinkMaxAdm &&
			sb.currentG > 16 {
			// Non-refill shrink: lower the target; seats leave via natural
			// exits and are not refilled until the band is re-entered.
			target := sb.currentG - sbGShrinkStep
			if float64(target)-rBar <= mHi {
				sb.shrinkStreak = 0
			}
			sb.currentG = target
			sb.lastGActionT = now
		}
	default:
		sb.shrinkStreak = 0
	}
}

// effectiveG returns the current group-size target.
func (sb *sbState) effectiveG() int {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	return sb.currentG
}

// knee returns the current warmth-EDF knee.
func (sb *sbState) knee() time.Duration {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	return sb.currentKnee
}

// stepTau runs one τ-controller readout + tracker update (call at most once
// per sbTauEvalEvery). kneeMax clamps the knee (deployment bound).
func (sb *sbState) stepTau(kneeMax time.Duration, now time.Time) {
	sb.mu.Lock()
	defer sb.mu.Unlock()

	// Cross-freeze: no τ action for sbTauCrossFreez after any G action.
	if !sb.lastGActionT.IsZero() && now.Before(sb.lastGActionT.Add(sbTauCrossFreez)) {
		return
	}

	// Validity gate: starved estimator holds the prior.
	if sb.sampleCount < sbTauMinSamples {
		return
	}

	// Plateau reference and threshold.
	plateau := sb.bins[0].ratio()
	if plateau <= 0 {
		return
	}
	theta := math.Max(plateau-0.20, 0.5*plateau)
	theta = math.Min(math.Max(theta, 0.15), 0.90)

	// Largest valid bin whose ratio clears θ; τ̂ is its upper edge,
	// log-interpolated with the next valid bin when it brackets the edge.
	tauHat := time.Duration(0)
	for i := 1; i < len(sb.bins)-1; i++ { // exclude plateau-ref and floor-ref
		if sb.bins[i].n >= sbBinMinSamples && sb.bins[i].ratio() >= theta {
			tauHat = sbBinUpperEdge(i)
		} else if sb.bins[i].n >= sbBinMinSamples {
			break
		}
	}
	if tauHat == 0 {
		return // no valid warm bin: hold
	}
	tauHat = time.Duration(math.Min(float64(tauHat), float64(kneeMax)))

	residual := math.Log2(float64(tauHat) / float64(sb.currentKnee))
	dir := 0
	if residual > 0 {
		dir = 1
	} else if residual < 0 {
		dir = -1
	}

	// Flip latch: a large confirmed residual jumps (bounded) and locks
	// direction, killing boundary flip-flopping.
	if math.Abs(residual) >= sbTauFlipSignal {
		sb.flipStreak++
		if sb.flipStreak >= 2 && dir != 0 {
			jump := sb.currentKnee
			if dir > 0 {
				jump = time.Duration(math.Min(float64(sb.currentKnee)*2, float64(tauHat)))
			} else {
				jump = time.Duration(math.Max(float64(sb.currentKnee)/2, float64(tauHat)))
			}
			sb.currentKnee = clampKnee(jump, kneeMax)
			sb.latchDir = dir
			sb.latchUntil = now.Add(sbTauLatchLock)
			sb.flipStreak = 0
			sb.evalStreak = 0
			return
		}
	} else {
		sb.flipStreak = 0
	}

	// Directional lock: while latched, only moves in the latched direction.
	if !sb.latchUntil.IsZero() && now.Before(sb.latchUntil) && dir != 0 && dir != sb.latchDir {
		return
	}

	// Deadband + geometric step.
	if math.Abs(residual) > sbTauDeadband {
		sb.evalStreak++
		if sb.evalStreak >= 2 {
			next := sb.currentKnee
			if dir > 0 {
				next = time.Duration(float64(sb.currentKnee) * sbTauStep)
			} else {
				next = time.Duration(float64(sb.currentKnee) / sbTauStep)
			}
			sb.currentKnee = clampKnee(next, kneeMax)
			sb.evalStreak = 0
		}
	} else {
		sb.evalStreak = 0
	}
}

// freezeTauAfterGAction applies the cross-freeze interlock (kept for
// symmetry with the design doc; stepTau reads lastGActionT directly).
func (sb *sbState) freezeTauAfterGAction() {}

func sbBinUpperEdge(i int) time.Duration {
	// Bins: 0=[0,30) 1=[30,60) 2=[60,120) 3=[120,240) 4=[240,480) 5=[480,inf).
	// Upper edges for bins 1..4 are 60, 120, 240, 480 seconds.
	edges := [...]time.Duration{60 * time.Second, 120 * time.Second, 240 * time.Second, 480 * time.Second}
	if i >= 1 && i-1 < len(edges) {
		return edges[i-1]
	}
	return 480 * time.Second
}

func clampKnee(v, kneeMax time.Duration) time.Duration {
	if v < sbKneeFloor {
		return sbKneeFloor
	}
	if v > kneeMax {
		return kneeMax
	}
	return v
}
