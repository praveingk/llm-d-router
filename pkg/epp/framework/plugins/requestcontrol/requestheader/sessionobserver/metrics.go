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

package sessionobserver

import (
	"errors"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// sessionTurnsBuckets is the per-session turn-count ladder, from a
// single-shot question to a long agent session.
var sessionTurnsBuckets = []float64{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}

// sessionDurationBuckets is the per-session active-duration ladder, from a
// one-minute exchange to a day-long working session.
var sessionDurationBuckets = []float64{1, 5, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400}

// observerMetrics holds the plugin's Prometheus collectors. Every collector
// uses the llm_d_epp subsystem, so the fully qualified names are
// llm_d_epp_<name> — the prefix the EPP's dashboards and alerts expect.
type observerMetrics struct {
	requests       *prometheus.CounterVec
	requestBytes   *prometheus.CounterVec
	started        *prometheus.CounterVec
	turns          *prometheus.HistogramVec
	duration       *prometheus.HistogramVec
	evictions      *prometheus.CounterVec
	sessionsActive *prometheus.GaugeVec
	tableSize      prometheus.Gauge

	// activeMu guards the sessionsActive series bookkeeping below, so the
	// incremental request-path updates and the sweep's full recompute stay
	// consistent and vanished series are deleted rather than going stale.
	activeMu    sync.Mutex
	activeKnown map[string]map[string]struct{}
}

func newObserverMetrics() *observerMetrics {
	return &observerMetrics{
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "session_requests_total",
			Help: metricsutil.HelpMsgWithStability(
				"Observed requests by resolved client harness and session header, from the passive session-observer plugin.",
				compbasemetrics.ALPHA),
		}, []string{"harness", "session_header"}),
		requestBytes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "session_request_bytes_total",
			Help: metricsutil.HelpMsgWithStability(
				"Observed request body bytes by resolved client harness and session header, from the passive session-observer plugin.",
				compbasemetrics.ALPHA),
		}, []string{"harness", "session_header"}),
		started: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "sessions_started_total",
			Help: metricsutil.HelpMsgWithStability(
				"Sessions newly tracked by the passive session-observer plugin, by the harness of their first request.",
				compbasemetrics.ALPHA),
		}, []string{"harness"}),
		turns: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "session_turns",
			Help: metricsutil.HelpMsgWithStability(
				"Turns per evicted session, by the harness fixed at the session's first request.",
				compbasemetrics.ALPHA),
			Buckets: sessionTurnsBuckets,
		}, []string{"harness"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "session_duration_seconds",
			Help: metricsutil.HelpMsgWithStability(
				"Active duration (last request minus first request) per evicted session, by harness.",
				compbasemetrics.ALPHA),
			Buckets: sessionDurationBuckets,
		}, []string{"harness"}),
		evictions: prometheus.NewCounterVec(prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "session_table_evictions_total",
			Help: metricsutil.HelpMsgWithStability(
				"Session-table evictions by reason: idle (past the idle timeout), capacity (least recently active idle session dropped for a new one), expired (in-flight request entry dropped without a completion hook).",
				compbasemetrics.ALPHA),
		}, []string{"reason"}),
		sessionsActive: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "sessions_active",
			Help: metricsutil.HelpMsgWithStability(
				"Tracked sessions by harness and activity window: \"inflight\" counts sessions with at least one request dispatched to a model server and not yet finished (requests queued in flow control are not in flight); each configured window counts sessions whose last request arrived within it.",
				compbasemetrics.ALPHA),
		}, []string{"harness", "window"}),
		tableSize: prometheus.NewGauge(prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "session_table_size",
			Help: metricsutil.HelpMsgWithStability(
				"Sessions currently tracked by the passive session-observer plugin.",
				compbasemetrics.ALPHA),
		}),
		activeKnown: make(map[string]map[string]struct{}),
	}
}

// register adds every collector to reg. Registration goes through the
// handle's MetricsRecorder — the same registry the EPP serves on :9090 — so
// the collectors are visible on the /metrics endpoint. registerOrReuse
// tolerates a collector that an earlier instance of the plugin already
// registered on the shared registry.
func (m *observerMetrics) register(reg prometheus.Registerer) error {
	return errors.Join(
		registerOrReuse(reg, &m.requests),
		registerOrReuse(reg, &m.requestBytes),
		registerOrReuse(reg, &m.started),
		registerOrReuse(reg, &m.turns),
		registerOrReuse(reg, &m.duration),
		registerOrReuse(reg, &m.evictions),
		registerOrReuse(reg, &m.sessionsActive),
		registerOrReuse(reg, &m.tableSize),
	)
}

// registerOrReuse registers c on reg, reusing an already-registered
// collector of the same type when present.
func registerOrReuse[C prometheus.Collector](reg prometheus.Registerer, target *C) error {
	err := reg.Register(*target)
	if err == nil {
		return nil
	}
	var already prometheus.AlreadyRegisteredError
	if errors.As(err, &already) {
		if existing, ok := already.ExistingCollector.(C); ok {
			*target = existing
			return nil
		}
	}
	return err
}

// addActive adjusts one sessions_active series by delta, remembering it so
// a later recompute can delete it when the series disappears.
func (m *observerMetrics) addActive(harness, window string, delta float64) {
	m.sessionsActive.WithLabelValues(harness, window).Add(delta)
	m.activeMu.Lock()
	defer m.activeMu.Unlock()
	windows, ok := m.activeKnown[harness]
	if !ok {
		windows = make(map[string]struct{})
		m.activeKnown[harness] = windows
	}
	windows[window] = struct{}{}
}

// recomputeActive replaces every sessions_active series with the freshly
// computed snapshot and deletes series that no longer exist, so harnesses
// whose last session was evicted disappear instead of freezing at their
// final value.
func (m *observerMetrics) recomputeActive(active map[string]map[string]int) {
	m.activeMu.Lock()
	defer m.activeMu.Unlock()
	for harness, windows := range active {
		for window, count := range windows {
			m.sessionsActive.WithLabelValues(harness, window).Set(float64(count))
			known, ok := m.activeKnown[harness]
			if !ok {
				known = make(map[string]struct{})
				m.activeKnown[harness] = known
			}
			known[window] = struct{}{}
		}
	}
	for harness, windows := range m.activeKnown {
		for window := range windows {
			if _, still := active[harness][window]; !still {
				m.sessionsActive.DeleteLabelValues(harness, window)
				delete(windows, window)
			}
		}
		if len(windows) == 0 {
			delete(m.activeKnown, harness)
		}
	}
}
