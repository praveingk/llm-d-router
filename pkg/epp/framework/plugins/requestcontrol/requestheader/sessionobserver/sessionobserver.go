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

// Package sessionobserver provides a passive RequestHeaderProcessor plugin
// that records per-session usage metrics without influencing any request
// decision.
//
// The plugin resolves a session identifier from a priority list of agent
// session headers and a client-harness identifier from user-agent style
// headers, and maintains a bounded table of per-session counters (turns,
// first/last seen, in-flight requests). It never writes a request attribute,
// never mutates headers, and never takes part in filtering, scoring or
// admission: it only reads request.Headers and the request body size. Raw
// session IDs are never emitted; the table is keyed by a salted SHA-256 hash
// truncated to 16 bytes, so metric output cannot be joined back to a client
// session.
package sessionobserver

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PluginType is the type string used to select this plugin in the EPP config.
const PluginType = "session-observer"

// Label values for the session_header label. A request whose session header
// carries a value that is oversized or non-printable is counted but never
// tracked as a session.
const (
	sessionHeaderNone    = "none"
	sessionHeaderInvalid = "invalid"
)

// Harness label values for requests that carry no harness header at all
// ("unknown") and for values that are unparseable or beyond the label cap
// ("other"; the BoundedLabel overflow value).
const (
	harnessUnknown = "unknown"
	harnessOther   = metrics.OverflowValue
)

// maxHarnessTokenLength bounds the parsed harness product token.
const maxHarnessTokenLength = 64

// inflightExpiryTTL bounds how long an in-flight request entry may live
// without a completion. The ResponseBody hook fires for every request that
// picked a pod — including failed, aborted, evicted and disconnected streams —
// but requests rejected before a pod was picked (admission denial, no
// endpoints, scheduler error) have no completion hook at this commit. Those
// entries are expired by the sweep and counted under the "expired" eviction
// reason instead of leaking forever.
const inflightExpiryTTL = 900 * time.Second

// windowInflight is the sessions_active window label counting sessions with
// at least one in-flight request.
const windowInflight = "inflight"

// defaultSessionHeaders is the built-in ordered list, checked first-match
// wins. Request headers are stored with lowercased keys (see
// handlers.HandleRequestHeaders), so the list must be lowercase to match.
var defaultSessionHeaders = []string{
	"x-claude-code-session-id",
	"x-opencode-session-id",
	"x-session-id",
	"session-id",
	"session_id",
	"x-session-affinity",
}

// defaultHarnessHeaders is the ordered list of headers a client harness is
// parsed from.
var defaultHarnessHeaders = []string{
	"x-client-user-agent",
	"user-agent",
}

// defaultActiveWindowsSeconds lists the sessions_active window sizes.
var defaultActiveWindowsSeconds = []int{300, 3600}

// Parameters is the user-facing plugin configuration block. Every field is
// optional; absent fields keep the defaults below.
type Parameters struct {
	// SessionHeaders is the priority-ordered list of session-ID headers. The
	// first header whose trimmed value is non-empty supplies the session ID;
	// its lowercased name becomes the session_header label.
	SessionHeaders []string `json:"sessionHeaders,omitempty"`
	// HarnessHeaders is the priority-ordered list of client-harness headers.
	HarnessHeaders []string `json:"harnessHeaders,omitempty"`
	// MaxHarnessLabels caps the distinct harness label values.
	MaxHarnessLabels int `json:"maxHarnessLabels,omitempty"`
	// ActiveWindowsSeconds lists the sessions_active window sizes in seconds.
	ActiveWindowsSeconds []int `json:"activeWindowsSeconds,omitempty"`
	// SessionIdleTimeoutSeconds is how long a session with no in-flight
	// request is kept after its last request before eviction.
	SessionIdleTimeoutSeconds int `json:"sessionIdleTimeoutSeconds,omitempty"`
	// MaxSessions caps the tracked-session table size.
	MaxSessions int `json:"maxSessions,omitempty"`
	// GaugeRefreshSeconds is the background sweep interval: idle/capacity
	// eviction and gauge recompute cadence.
	GaugeRefreshSeconds int `json:"gaugeRefreshSeconds,omitempty"`
	// MaxSessionIdBytes bounds the session-ID length; longer values count as
	// session_header="invalid" and are not tracked.
	MaxSessionIdBytes int `json:"maxSessionIdBytes,omitempty"`
}

// defaultParameters returns a fresh Parameters value with slice fields
// copied from the package-level defaults, so a later json.Decode into the
// struct can never mutate the shared defaults.
func defaultParameters() Parameters {
	return Parameters{
		SessionHeaders:            append([]string(nil), defaultSessionHeaders...),
		HarnessHeaders:            append([]string(nil), defaultHarnessHeaders...),
		MaxHarnessLabels:          50,
		ActiveWindowsSeconds:      append([]int(nil), defaultActiveWindowsSeconds...),
		SessionIdleTimeoutSeconds: 3600,
		MaxSessions:               100000,
		GaugeRefreshSeconds:       15,
		MaxSessionIdBytes:         256,
	}
}

// resolve validates the parameters and returns the effective configuration.
// A nil (absent) list falls back to the built-in defaults; an explicitly
// empty list is rejected.
func (p Parameters) resolve() (Parameters, error) {
	if p.SessionHeaders == nil {
		p.SessionHeaders = defaultSessionHeaders
	}
	if p.HarnessHeaders == nil {
		p.HarnessHeaders = defaultHarnessHeaders
	}
	if p.ActiveWindowsSeconds == nil {
		p.ActiveWindowsSeconds = defaultActiveWindowsSeconds
	}
	p.SessionHeaders = lowerAll(p.SessionHeaders)
	p.HarnessHeaders = lowerAll(p.HarnessHeaders)

	if len(p.SessionHeaders) == 0 {
		return p, fmt.Errorf("sessionHeaders must not be empty")
	}
	if len(p.HarnessHeaders) == 0 {
		return p, fmt.Errorf("harnessHeaders must not be empty")
	}
	if p.MaxHarnessLabels <= 0 {
		return p, fmt.Errorf("maxHarnessLabels must be > 0, got %d", p.MaxHarnessLabels)
	}
	if p.SessionIdleTimeoutSeconds <= 0 {
		return p, fmt.Errorf("sessionIdleTimeoutSeconds must be > 0, got %d", p.SessionIdleTimeoutSeconds)
	}
	if p.MaxSessions <= 0 {
		return p, fmt.Errorf("maxSessions must be > 0, got %d", p.MaxSessions)
	}
	if p.GaugeRefreshSeconds <= 0 {
		return p, fmt.Errorf("gaugeRefreshSeconds must be > 0, got %d", p.GaugeRefreshSeconds)
	}
	if p.MaxSessionIdBytes <= 0 {
		return p, fmt.Errorf("maxSessionIdBytes must be > 0, got %d", p.MaxSessionIdBytes)
	}
	for i, w := range p.ActiveWindowsSeconds {
		if w <= 0 {
			return p, fmt.Errorf("activeWindowsSeconds[%d] must be > 0, got %d", i, w)
		}
	}
	return p, nil
}

// lowerAll lowercases a copy of the header list so entries match the
// lowercased request-header map keys.
func lowerAll(headers []string) []string {
	out := make([]string, len(headers))
	for i, h := range headers {
		out[i] = strings.ToLower(strings.TrimSpace(h))
	}
	return out
}

// windowDurations converts the configured windows to durations.
func (p Parameters) windowDurations() []time.Duration {
	out := make([]time.Duration, 0, len(p.ActiveWindowsSeconds))
	for _, w := range p.ActiveWindowsSeconds {
		out = append(out, time.Duration(w)*time.Second)
	}
	return out
}

// formatWindow renders a window size as a compact label: whole hours as
// "Nh", whole minutes as "Nm", anything else as "Ns" (300→"5m", 3600→"1h",
// 90→"90s").
func formatWindow(seconds int) string {
	switch {
	case seconds%3600 == 0:
		return fmt.Sprintf("%dh", seconds/3600)
	case seconds%60 == 0:
		return fmt.Sprintf("%dm", seconds/60)
	default:
		return fmt.Sprintf("%ds", seconds)
	}
}

// Plugin is the passive session-observer plugin.
type Plugin struct {
	typedName plugin.TypedName
	params    Parameters

	table   *sessionTable
	metrics *observerMetrics

	// harnessLabels bounds the distinct harness label values.
	harnessLabels *metrics.BoundedLabel

	// now is injectable for tests.
	now func() time.Time
}

// PluginFactory is the factory function for the session-observer plugin. It
// registers the plugin's collectors on the handle's metrics recorder — the
// same registry the EPP serves on :9090 — and starts the background sweep,
// which stops when the handle's context is cancelled.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := defaultParameters()
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("session-observer: failed to decode parameters: %w", err)
		}
	}
	params, err := params.resolve()
	if err != nil {
		return nil, fmt.Errorf("session-observer: invalid parameters: %w", err)
	}

	harnessLabels := metrics.NewBoundedLabel(params.MaxHarnessLabels)
	harnessLabels.Pin(harnessUnknown)
	harnessLabels.Pin(harnessOther)

	p := &Plugin{
		typedName:     plugin.TypedName{Type: PluginType, Name: name},
		params:        params,
		table:         newSessionTable(params.MaxSessions, time.Duration(params.SessionIdleTimeoutSeconds)*time.Second, inflightExpiryTTL),
		metrics:       newObserverMetrics(),
		harnessLabels: harnessLabels,
		now:           time.Now,
	}

	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := p.metrics.register(reg); err != nil {
				return nil, fmt.Errorf("session-observer: failed to register metrics: %w", err)
			}
		}
		go p.runSweep(handle.Context(), time.Duration(params.GaugeRefreshSeconds)*time.Second)
	}
	return p, nil
}

// compile-time interface assertions. The plugin deliberately does not
// implement plugin.ProducerPlugin: it writes no request attribute and
// declares no data dependency, so the data layer treats it as fully passive.
var (
	_ requestcontrol.RequestHeaderProcessor = &Plugin{}
	_ requestcontrol.ResponseBodyProcessor  = &Plugin{}
)

// TypedName returns the plugin's type and instance name.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// RequestHeader runs after the InferenceRequest is assembled and before
// admission control — the earliest request hook, the same one the
// agent-identity plugin uses. It reads headers and the request body size,
// counts the request, and tracks the session. It never mutates the request.
func (p *Plugin) RequestHeader(_ context.Context, request *scheduling.InferenceRequest) error {
	if request == nil {
		return nil
	}
	now := p.now()

	headerLabel, sessionID := resolveSessionID(request.Headers, p.params.SessionHeaders, p.params.MaxSessionIdBytes)
	harness := p.resolveHarness(request.Headers)

	p.metrics.requests.WithLabelValues(harness, headerLabel).Inc()
	if request.RequestSizeBytes > 0 {
		p.metrics.requestBytes.WithLabelValues(harness, headerLabel).Add(float64(request.RequestSizeBytes))
	}

	if sessionID == "" {
		// "none" (no session header matched) or "invalid" (oversized or
		// non-printable value): counted above, not tracked below.
		return nil
	}

	hash := p.table.hashSession(sessionID)
	sessHarness, created, inflightAfter := p.table.track(hash, harness, request.RequestID, now, p.onEvict)
	if created {
		p.metrics.started.WithLabelValues(sessHarness).Inc()
	}
	if inflightAfter == 1 {
		// 0 -> 1 transition: the session now has an in-flight request.
		p.metrics.addActive(sessHarness, windowInflight, 1)
	}
	return nil
}

// ResponseBody is the completion hook. The director invokes it with
// EndOfStream=true exactly once per request that picked a pod — including
// streams that failed, were aborted, were evicted or were torn down by a
// client disconnect (TerminationCause carries which). The decrement here is
// idempotent; requests that never picked a pod have no completion hook and
// are expired by the sweep instead.
func (p *Plugin) ResponseBody(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	if response == nil || !response.EndOfStream {
		return
	}
	requestID := response.RequestID
	if requestID == "" && request != nil {
		requestID = request.RequestID
	}
	if requestID == "" {
		return
	}
	harness, inflightAfter, ok := p.table.complete(requestID)
	if ok && inflightAfter == 0 {
		p.metrics.addActive(harness, windowInflight, -1)
	}
}

// resolveSessionID walks the priority list and returns the session_header
// label plus the raw session ID (empty when the request is not trackable).
// The label is the matched header's name, "none" when nothing matched, or
// "invalid" when the matched value is longer than maxSessionIdBytes or
// contains non-printable characters.
func resolveSessionID(headers map[string]string, priority []string, maxSessionIDBytes int) (label, id string) {
	for _, header := range priority {
		value, ok := headers[header]
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if len(value) > maxSessionIDBytes || !printableASCII(value) {
			return sessionHeaderInvalid, ""
		}
		return header, value
	}
	return sessionHeaderNone, ""
}

// printableASCII reports whether s is valid UTF-8 consisting solely of
// printable characters (letters, marks, numbers, punctuation, symbols and
// spaces). Control characters and invalid encodings make a session ID
// untrustworthy as a label-adjacent value, so such requests count as
// "invalid" and are not tracked.
func printableASCII(s string) bool {
	if !utf8.ValidString(s) {
		return false
	}
	for _, r := range s {
		if !unicode.IsPrint(r) {
			return false
		}
	}
	return true
}

// resolveHarness parses the client harness from the first non-empty harness
// header and bounds its label value.
func (p *Plugin) resolveHarness(headers map[string]string) string {
	for _, header := range p.params.HarnessHeaders {
		value, ok := headers[header]
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		return p.harnessLabels.Bound(parseHarnessToken(value))
	}
	return harnessUnknown
}

// parseHarnessToken extracts the product token from a user-agent style
// value: the run before the first "/" or space, lowercased and trimmed. The
// token must be at most 64 characters of [a-z0-9._-]; anything else yields
// "other". Examples: "claude-cli/2.1.3 (external, cli)" -> "claude-cli",
// "opencode/1.0.2" -> "opencode", "codex_cli_rs/0.131.0 (...)" ->
// "codex_cli_rs".
func parseHarnessToken(value string) string {
	value = strings.TrimSpace(value)
	if i := strings.IndexAny(value, "/ "); i >= 0 {
		value = value[:i]
	}
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" || len(value) > maxHarnessTokenLength {
		return harnessOther
	}
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
		default:
			return harnessOther
		}
	}
	return value
}

// runSweep evicts stale sessions and recomputes gauges every interval until
// ctx is cancelled.
func (p *Plugin) runSweep(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.sweepOnce(p.now())
		}
	}
}

// sweepOnce runs one sweep pass: expire orphaned in-flight request entries,
// evict idle sessions, then recompute all gauge series from the table.
func (p *Plugin) sweepOnce(now time.Time) {
	p.table.sweep(now, p.onEvict)
	p.recomputeGauges(now)
}

// onEvict records a table eviction. Session evictions (idle, capacity) also
// observe the terminal turns and duration histograms; "expired" evictions
// are in-flight request entries whose session may still be tracked.
func (p *Plugin) onEvict(e eviction) {
	p.metrics.evictions.WithLabelValues(e.reason).Inc()
	if e.tracked {
		p.metrics.turns.WithLabelValues(e.harness).Observe(float64(e.turns))
		p.metrics.duration.WithLabelValues(e.harness).Observe(e.duration.Seconds())
	}
}

// recomputeGauges rebuilds every sessions_active series from a table
// snapshot and drops series that no longer exist, so harnesses whose last
// session was evicted disappear instead of going stale at their last value.
func (p *Plugin) recomputeGauges(now time.Time) {
	active, size := p.table.snapshotWindows(now, p.params.windowDurations())
	p.metrics.recomputeActive(active)
	p.metrics.tableSize.Set(float64(size))
}

// hashKey computes the per-process salted session hash: the first 16 bytes
// of SHA-256(salt || session ID), hex-encoded. The salt is generated once
// per process so hashes cannot be correlated across restarts or computed
// offline by anyone who knows the ID format.
func (t *sessionTable) hashSession(id string) string {
	h := sha256.New()
	h.Write(t.salt)
	h.Write([]byte(id))
	return hex.EncodeToString(h.Sum(nil)[:16])
}

// newProcessSalt returns fresh random salt for the session hash.
func newProcessSalt() []byte {
	salt := make([]byte, 32)
	if _, err := rand.Read(salt); err != nil {
		// crypto/rand never fails on the supported platforms; a fixed salt
		// keeps the plugin serving rather than crashing the EPP.
		for i := range salt {
			salt[i] = byte(i * 7)
		}
	}
	return salt
}
