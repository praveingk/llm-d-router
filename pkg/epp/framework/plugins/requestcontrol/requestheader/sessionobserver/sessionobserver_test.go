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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// fakeClock gives tests deterministic control over the plugin's now.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// newTestPlugin builds a plugin from raw JSON parameters ("" for defaults),
// registers its collectors on a private registry for gathering, and switches
// it to a fake clock the test can advance.
func newTestPlugin(t *testing.T, paramsJSON string) (*Plugin, *fakeClock, *prometheus.Registry) {
	t.Helper()

	var decoder *json.Decoder
	if paramsJSON != "" {
		decoder = json.NewDecoder(strings.NewReader(paramsJSON))
	}
	pi, err := PluginFactory("test", decoder, nil)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	p := pi.(*Plugin)

	reg := prometheus.NewRegistry()
	if err := p.metrics.register(reg); err != nil {
		t.Fatalf("register metrics: %v", err)
	}
	clock := newFakeClock()
	p.now = clock.Now
	return p, clock, reg
}

func newRequest(id string, headers map[string]string, size int) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID:        id,
		Headers:          headers,
		RequestSizeBytes: size,
	}
}

func mustRequestHeader(t *testing.T, p *Plugin, req *scheduling.InferenceRequest) {
	t.Helper()
	if err := p.RequestHeader(context.Background(), req); err != nil {
		t.Fatalf("RequestHeader: %v", err)
	}
}

// mustDispatch runs the PreRequest hook, simulating the director having
// picked an endpoint for the request.
func mustDispatch(t *testing.T, p *Plugin, req *scheduling.InferenceRequest) {
	t.Helper()
	if err := p.PreRequest(context.Background(), req, nil); err != nil {
		t.Fatalf("PreRequest: %v", err)
	}
}

// fullTurn runs the complete happy-path hook sequence for a request:
// counted, dispatched, and settled.
func fullTurn(t *testing.T, p *Plugin, req *scheduling.InferenceRequest) {
	t.Helper()
	mustRequestHeader(t, p, req)
	mustDispatch(t, p, req)
	p.ResponseBody(context.Background(), req, &requestcontrol.Response{
		RequestID:   req.RequestID,
		EndOfStream: true,
	}, nil)
}

// settle completes a request through the plugin's completion hook.
func settle(t *testing.T, p *Plugin, requestID string) {
	t.Helper()
	p.ResponseBody(context.Background(), nil, &requestcontrol.Response{
		RequestID:   requestID,
		EndOfStream: true,
	}, nil)
}

// counterValue reads one labelled counter from the plugin's collectors.
func counterValue(t *testing.T, p *Plugin, name string, labels ...string) float64 {
	t.Helper()
	switch name {
	case "requests":
		return testutil.ToFloat64(p.metrics.requests.WithLabelValues(labels...))
	case "requestBytes":
		return testutil.ToFloat64(p.metrics.requestBytes.WithLabelValues(labels...))
	case "started":
		return testutil.ToFloat64(p.metrics.started.WithLabelValues(labels...))
	case "evictions":
		return testutil.ToFloat64(p.metrics.evictions.WithLabelValues(labels...))
	}
	t.Fatalf("unknown counter %q", name)
	return 0
}

func gaugeValue(t *testing.T, p *Plugin, harness, window string) float64 {
	t.Helper()
	return testutil.ToFloat64(p.metrics.sessionsActive.WithLabelValues(harness, window))
}

// gatherText renders every metric family in reg as text, labels included.
func gatherText(t *testing.T, reg *prometheus.Registry) string {
	t.Helper()
	var buf bytes.Buffer
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	for _, mf := range families {
		buf.WriteString(mf.GetName())
		for _, m := range mf.GetMetric() {
			buf.WriteString("{")
			for _, lp := range m.GetLabel() {
				buf.WriteString(lp.GetName() + "=" + lp.GetValue() + ",")
			}
			buf.WriteString("}\n")
		}
	}
	return buf.String()
}

func TestConfigDefaults(t *testing.T) {
	p, _, _ := newTestPlugin(t, "")
	got := p.params
	want := defaultParameters()
	if len(got.SessionHeaders) != len(want.SessionHeaders) || got.SessionHeaders[0] != "x-claude-code-session-id" {
		t.Errorf("default sessionHeaders = %v, want %v", got.SessionHeaders, want.SessionHeaders)
	}
	if len(got.HarnessHeaders) != 2 || got.HarnessHeaders[0] != "x-client-user-agent" {
		t.Errorf("default harnessHeaders = %v", got.HarnessHeaders)
	}
	if got.MaxHarnessLabels != 50 || got.SessionIdleTimeoutSeconds != 3600 ||
		got.MaxSessions != 100000 || got.GaugeRefreshSeconds != 15 || got.MaxSessionIdBytes != 256 {
		t.Errorf("defaults not applied: %+v", got)
	}
	if len(got.ActiveWindowsSeconds) != 2 || got.ActiveWindowsSeconds[0] != 300 || got.ActiveWindowsSeconds[1] != 3600 {
		t.Errorf("default activeWindowsSeconds = %v", got.ActiveWindowsSeconds)
	}
}

func TestConfigDecodeAndValidation(t *testing.T) {
	full := `{
		"sessionHeaders": ["x-first", "x-second"],
		"harnessHeaders": ["x-client-user-agent", "user-agent"],
		"maxHarnessLabels": 7,
		"activeWindowsSeconds": [90, 300, 3600],
		"sessionIdleTimeoutSeconds": 120,
		"maxSessions": 9,
		"gaugeRefreshSeconds": 3,
		"maxSessionIdBytes": 64
	}`
	p, _, _ := newTestPlugin(t, full)
	if p.params.SessionHeaders[0] != "x-first" || p.params.MaxHarnessLabels != 7 ||
		p.params.SessionIdleTimeoutSeconds != 120 || p.params.MaxSessions != 9 ||
		p.params.GaugeRefreshSeconds != 3 || p.params.MaxSessionIdBytes != 64 {
		t.Errorf("parameters not decoded: %+v", p.params)
	}
	if len(p.params.ActiveWindowsSeconds) != 3 || p.params.ActiveWindowsSeconds[0] != 90 {
		t.Errorf("activeWindowsSeconds not decoded: %v", p.params.ActiveWindowsSeconds)
	}

	tests := []struct {
		name   string
		params string
	}{
		{"zero maxHarnessLabels", `{"maxHarnessLabels": 0}`},
		{"negative maxSessions", `{"maxSessions": -1}`},
		{"zero idle timeout", `{"sessionIdleTimeoutSeconds": 0}`},
		{"zero gauge refresh", `{"gaugeRefreshSeconds": 0}`},
		{"zero maxSessionIdBytes", `{"maxSessionIdBytes": 0}`},
		{"zero window", `{"activeWindowsSeconds": [300, 0]}`},
		{"empty sessionHeaders", `{"sessionHeaders": []}`},
		{"empty harnessHeaders", `{"harnessHeaders": []}`},
		{"bad json", `{`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PluginFactory("bad", json.NewDecoder(strings.NewReader(tt.params)), nil)
			if err == nil {
				t.Fatalf("expected error for %s", tt.params)
			}
			if !strings.Contains(err.Error(), "session-observer") {
				t.Errorf("error should name the plugin: %v", err)
			}
		})
	}
}

func TestFormatWindow(t *testing.T) {
	tests := []struct {
		seconds int
		want    string
	}{
		{300, "5m"}, {3600, "1h"}, {90, "90s"}, {7200, "2h"}, {45, "45s"}, {2700, "45m"},
	}
	for _, tt := range tests {
		if got := formatWindow(tt.seconds); got != tt.want {
			t.Errorf("formatWindow(%d) = %q, want %q", tt.seconds, got, tt.want)
		}
	}
}

func TestRequestHeaderCounting(t *testing.T) {
	p, _, _ := newTestPlugin(t, "")

	cases := []struct {
		name        string
		headers     map[string]string
		wantHeader  string
		wantHarness string
		size        int
		wantBytes   float64
	}{
		{"claude header", map[string]string{"x-claude-code-session-id": "s1"}, "x-claude-code-session-id", "unknown", 100, 100},
		{"opencode header", map[string]string{"x-opencode-session-id": "s2"}, "x-opencode-session-id", "unknown", 200, 200},
		{"x-session-id header", map[string]string{"x-session-id": "s3"}, "x-session-id", "unknown", 300, 300},
		{"session-id header", map[string]string{"session-id": "s4"}, "session-id", "unknown", 400, 400},
		{"legacy session_id header", map[string]string{"session_id": "s5"}, "session_id", "unknown", 500, 500},
		{"affinity header", map[string]string{"x-session-affinity": "s6"}, "x-session-affinity", "unknown", 600, 600},
		{"no session header", map[string]string{"user-agent": "opencode/1.0.2"}, "none", "opencode", 700, 700},
		// Both invalid cases share the {unknown, invalid} series, so the
		// second accumulates on top of the first.
		{"oversized value", map[string]string{"x-session-id": strings.Repeat("x", 257)}, "invalid", "unknown", 800, 800},
		{"non-printable value", map[string]string{"x-session-id": "bad\x01id"}, "invalid", "unknown", 900, 1700},
	}
	invalidSeen := 0
	for _, tt := range cases {
		req := newRequest("req-"+tt.name, tt.headers, tt.size)
		mustRequestHeader(t, p, req)

		wantCount := float64(1)
		if tt.wantHeader == "invalid" {
			invalidSeen++
			wantCount = float64(invalidSeen)
		}
		if got := counterValue(t, p, "requests", tt.wantHarness, tt.wantHeader); got != wantCount {
			t.Errorf("%s: requests{%s,%s} = %v, want %v", tt.name, tt.wantHarness, tt.wantHeader, got, wantCount)
		}
		if got := counterValue(t, p, "requestBytes", tt.wantHarness, tt.wantHeader); got != tt.wantBytes {
			t.Errorf("%s: requestBytes = %v, want %v", tt.name, got, tt.wantBytes)
		}
	}

	// A second request on the same header accumulates.
	mustRequestHeader(t, p, newRequest("req-again", map[string]string{"x-session-id": "s3"}, 50))
	if got := counterValue(t, p, "requests", "unknown", "x-session-id"); got != 2 {
		t.Errorf("accumulation: got %v, want 2", got)
	}
	if got := counterValue(t, p, "requestBytes", "unknown", "x-session-id"); got != 350 {
		t.Errorf("byte accumulation: got %v, want 350", got)
	}

	// "none" and "invalid" requests are counted but never tracked.
	if got := p.table.len(); got != 6 {
		t.Errorf("tracked sessions = %d, want 6 (none/invalid excluded)", got)
	}
}

func TestHeaderPriority(t *testing.T) {
	p, _, _ := newTestPlugin(t, "")

	// All defaults present at once: the first in the built-in priority
	// order wins.
	all := map[string]string{
		"x-claude-code-session-id": "claude",
		"x-opencode-session-id":    "oc",
		"x-session-id":             "xs",
		"session-id":               "sid",
		"session_id":               "usid",
		"x-session-affinity":       "aff",
	}
	mustRequestHeader(t, p, newRequest("r1", all, 10))
	if got := counterValue(t, p, "requests", "unknown", "x-claude-code-session-id"); got != 1 {
		t.Errorf("priority: claude header should win, requests = %v", got)
	}

	// Custom order overrides the built-in priority.
	p2, _, _ := newTestPlugin(t, `{"sessionHeaders": ["session-id", "x-claude-code-session-id"]}`)
	mustRequestHeader(t, p2, newRequest("r2", all, 10))
	if got := counterValue(t, p2, "requests", "unknown", "session-id"); got != 1 {
		t.Errorf("custom priority: session-id should win, requests = %v", got)
	}

	// Header present but blank: skipped, the next one wins.
	p3, _, _ := newTestPlugin(t, "")
	mustRequestHeader(t, p3, newRequest("r3", map[string]string{
		"x-claude-code-session-id": "   ",
		"session-id":               "sid",
	}, 10))
	if got := counterValue(t, p3, "requests", "unknown", "session-id"); got != 1 {
		t.Errorf("blank value should fall through to session-id, got requests{session-id} = %v", got)
	}
}

func TestParseHarnessToken(t *testing.T) {
	tests := []struct {
		value string
		want  string
	}{
		{"claude-cli/2.1.3 (external, cli)", "claude-cli"},
		{"opencode/1.0.2", "opencode"},
		{"codex_cli_rs/0.131.0 (Mac OS 15.4.0; arm64)", "codex_cli_rs"},
		{"Mozilla/5.0 (Macintosh)", "mozilla"},
		{"curl/8.4.0", "curl"},
		{"python-requests/2.31.0", "python-requests"},
		{"Claude-CLI/2.1.3", "claude-cli"}, // lowercased
		{"bare-agent", "bare-agent"},       // no version part
		{"  spaced/1.0  ", "spaced"},       // trimmed
		{"junk!!!/1.0", "other"},           // invalid characters
		{"!!!", "other"},
		{strings.Repeat("a", 65) + "/1.0", "other"}, // too long
		{"/1.0", "other"},                           // empty product token
	}
	for _, tt := range tests {
		if got := parseHarnessToken(tt.value); got != tt.want {
			t.Errorf("parseHarnessToken(%q) = %q, want %q", tt.value, got, tt.want)
		}
	}
}

func TestHarnessResolution(t *testing.T) {
	// Cap 5 admits claude-cli, opencode, h1, h2, h3 before overflowing;
	// the pinned "unknown"/"other" values never consume cap slots.
	p, _, _ := newTestPlugin(t, `{"maxHarnessLabels": 5}`)

	// No harness header at all -> unknown.
	mustRequestHeader(t, p, newRequest("r1", map[string]string{"x-session-id": "s1"}, 10))
	if got := counterValue(t, p, "requests", "unknown", "x-session-id"); got != 1 {
		t.Errorf("missing harness header should count as unknown, got %v", got)
	}

	// x-client-user-agent wins over user-agent.
	mustRequestHeader(t, p, newRequest("r2", map[string]string{
		"x-session-id":        "s2",
		"x-client-user-agent": "claude-cli/2.1.3 (external, cli)",
		"user-agent":          "curl/8.0",
	}, 10))
	if got := counterValue(t, p, "requests", "claude-cli", "x-session-id"); got != 1 {
		t.Errorf("x-client-user-agent should win, got %v", got)
	}

	// Falls back to user-agent.
	mustRequestHeader(t, p, newRequest("r3", map[string]string{
		"x-session-id": "s3",
		"user-agent":   "opencode/1.0.2",
	}, 10))
	if got := counterValue(t, p, "requests", "opencode", "x-session-id"); got != 1 {
		t.Errorf("user-agent fallback failed, got %v", got)
	}

	// Distinct harness values beyond the cap collapse to other.
	for _, ua := range []string{"h1/1", "h2/1", "h3/1", "h4/1"} {
		mustRequestHeader(t, p, newRequest("r-"+ua, map[string]string{
			"x-session-id": "s-" + ua,
			"user-agent":   ua,
		}, 10))
	}
	if got := counterValue(t, p, "requests", "h4", "x-session-id"); got != 0 {
		t.Errorf("value beyond the cap should not be admitted, got %v", got)
	}
	if got := counterValue(t, p, "requests", "other", "x-session-id"); got != 1 {
		t.Errorf("value beyond the cap should collapse to other, got %v", got)
	}

	// A value that appears after the cap filled still maps to a previously
	// admitted value.
	mustRequestHeader(t, p, newRequest("r-again", map[string]string{
		"x-session-id": "s-again",
		"user-agent":   "h1/1",
	}, 10))
	if got := counterValue(t, p, "requests", "h1", "x-session-id"); got != 2 {
		t.Errorf("admitted value should stay admitted, got %v", got)
	}
}

func TestHarnessFixedAtFirstRequest(t *testing.T) {
	p, _, _ := newTestPlugin(t, "")

	turn := func(id, session, ua string) {
		fullTurn(t, p, newRequest(id, map[string]string{
			"x-session-id": session,
			"user-agent":   ua,
		}, 10))
	}

	turn("r1", "s1", "opencode/1.0.2")
	turn("r2", "s1", "claude-cli/2.1.3") // same session, different harness

	// Session-scoped metrics keep the first request's harness...
	if got := counterValue(t, p, "started", "opencode"); got != 1 {
		t.Errorf("sessions_started_total{opencode} = %v, want 1", got)
	}
	if got := counterValue(t, p, "started", "claude-cli"); got != 0 {
		t.Errorf("sessions_started_total{claude-cli} = %v, want 0", got)
	}
	// ...while per-request counters reflect each request's own harness.
	if got := counterValue(t, p, "requests", "opencode", "x-session-id"); got != 1 {
		t.Errorf("requests{opencode} = %v, want 1", got)
	}
	if got := counterValue(t, p, "requests", "claude-cli", "x-session-id"); got != 1 {
		t.Errorf("requests{claude-cli} = %v, want 1", got)
	}
	// Both turns landed on one session.
	if got := p.table.len(); got != 1 {
		t.Errorf("table size = %d, want 1", got)
	}
}

func TestActiveGaugesPerWindow(t *testing.T) {
	// Idle timeout large enough that no session is evicted during the test.
	p, clock, _ := newTestPlugin(t, `{"activeWindowsSeconds": [300, 3600], "sessionIdleTimeoutSeconds": 7200}`)

	// Ages at sweep time: d=5000s (outside 1h, inside the idle timeout),
	// c=1000s (outside 5m, inside 1h), b=100s (inside 5m), a=0s (in flight).
	req := func(id, session string) {
		fullTurn(t, p, newRequest(id, map[string]string{
			"x-session-id": session, "user-agent": "opencode/1",
		}, 10))
	}
	req("r-d", "d")
	clock.Advance(4000 * time.Second)
	req("r-c", "c")
	clock.Advance(900 * time.Second)
	req("r-b", "b")
	clock.Advance(100 * time.Second)
	// Dispatched but never settled: counts as in flight.
	inflightReq := newRequest("r-a", map[string]string{
		"x-session-id": "a", "user-agent": "opencode/1",
	}, 10)
	mustRequestHeader(t, p, inflightReq)
	mustDispatch(t, p, inflightReq)

	p.sweepOnce(clock.Now())

	if got := gaugeValue(t, p, "opencode", "inflight"); got != 1 {
		t.Errorf("sessions_active{inflight} = %v, want 1 (a)", got)
	}
	if got := gaugeValue(t, p, "opencode", "5m"); got != 2 {
		t.Errorf("sessions_active{5m} = %v, want 2 (a, b)", got)
	}
	if got := gaugeValue(t, p, "opencode", "1h"); got != 3 {
		t.Errorf("sessions_active{1h} = %v, want 3 (a, b, c)", got)
	}
	if got := testutil.ToFloat64(p.metrics.tableSize); got != 4 {
		t.Errorf("table_size = %v, want 4", got)
	}

	// A second sweep with no traffic keeps the same values, and a series
	// whose sessions all vanished is dropped rather than frozen: evict
	// everything by advancing past the idle timeout and recompute.
	clock.Advance(7201 * time.Second)
	p.sweepOnce(clock.Now())
	if got := testutil.ToFloat64(p.metrics.tableSize); got != 0 {
		t.Errorf("table_size after full idle eviction = %v, want 0", got)
	}
	if got := counterValue(t, p, "evictions", "idle"); got != 4 {
		t.Errorf("idle evictions = %v, want 4", got)
	}
	if got := testutil.CollectAndCount(p.metrics.sessionsActive); got != 0 {
		t.Errorf("sessions_active series after full eviction = %d, want 0 (all deleted)", got)
	}
}

func TestInflightUpDown(t *testing.T) {
	p, _, _ := newTestPlugin(t, "")

	// A request that only reached RequestHeader — rejected before dispatch
	// by flow control (429), admission, or endpoint selection — is counted
	// but never in flight, and leaves nothing for the expiry backstop to
	// clean up.
	rejected := newRequest("r-rejected", map[string]string{"x-session-id": "s-rej"}, 10)
	mustRequestHeader(t, p, rejected)
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 0 {
		t.Errorf("inflight gauge after a rejected request = %v, want 0", got)
	}
	if got := p.table.inflightLen(); got != 0 {
		t.Errorf("in-flight entries after a rejected request = %d, want 0", got)
	}
	if got := counterValue(t, p, "requests", "unknown", "x-session-id"); got != 1 {
		t.Errorf("rejected request should still be counted, got %v", got)
	}
	if got := counterValue(t, p, "started", "unknown"); got != 1 {
		t.Errorf("rejected request should still create its session, got %v", got)
	}

	r1 := newRequest("r1", map[string]string{"x-session-id": "s1"}, 10)
	mustRequestHeader(t, p, r1)
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 0 {
		t.Errorf("inflight gauge after RequestHeader only = %v, want 0 (not dispatched yet)", got)
	}
	mustDispatch(t, p, r1)
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 1 {
		t.Errorf("inflight gauge after dispatch = %v, want 1", got)
	}

	// Second request on the same session: still one session in flight.
	r2 := newRequest("r2", map[string]string{"x-session-id": "s1"}, 10)
	mustRequestHeader(t, p, r2)
	mustDispatch(t, p, r2)
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 1 {
		t.Errorf("inflight gauge after second dispatch = %v, want 1", got)
	}

	// A repeated PreRequest for the same request ID must not double-count.
	mustDispatch(t, p, r2)
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 1 {
		t.Errorf("inflight gauge after duplicate dispatch = %v, want 1", got)
	}
	if got := p.table.inflightLen(); got != 2 {
		t.Errorf("in-flight entries after duplicate dispatch = %d, want 2", got)
	}

	// Intermediate chunks do not decrement.
	p.ResponseBody(context.Background(), nil, &requestcontrol.Response{RequestID: "r1", EndOfStream: false}, nil)
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 1 {
		t.Errorf("inflight gauge after intermediate chunk = %v, want 1", got)
	}

	// Every termination cause settles the request: natural, error,
	// client-disconnect and evicted streams all fire EndOfStream=true.
	for _, cause := range []requestcontrol.TerminationCause{
		requestcontrol.TerminationCauseNatural,
		requestcontrol.TerminationCauseError,
		requestcontrol.TerminationCauseClientDisconnect,
		requestcontrol.TerminationCauseEvicted,
	} {
		req := newRequest("r-cause", map[string]string{"x-session-id": "s-cause"}, 10)
		mustRequestHeader(t, p, req)
		mustDispatch(t, p, req)
		p.ResponseBody(context.Background(), req, &requestcontrol.Response{
			RequestID:        "r-cause",
			EndOfStream:      true,
			TerminationCause: cause,
		}, nil)
		if got := gaugeValue(t, p, "unknown", "inflight"); got != 1 {
			t.Errorf("inflight gauge after %s completion = %v, want 1", cause, got)
		}
	}

	// Settle r1 and r2: the session leaves the in-flight gauge once.
	settle(t, p, "r1")
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 1 {
		t.Errorf("inflight gauge after first completion = %v, want 1", got)
	}
	settle(t, p, "r2")
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 0 {
		t.Errorf("inflight gauge after both completions = %v, want 0", got)
	}

	// Double completion is a no-op, as is completing an unknown request.
	settle(t, p, "r2")
	settle(t, p, "never-existed")
	if got := gaugeValue(t, p, "unknown", "inflight"); got != 0 {
		t.Errorf("inflight gauge after idempotent completions = %v, want 0", got)
	}
	if got := p.table.inflightLen(); got != 0 {
		t.Errorf("in-flight entries = %d, want 0", got)
	}
}

func TestInflightExpiry(t *testing.T) {
	// A dispatched request whose completion never fires — the backstop
	// case, e.g. a teardown the EPP never observes — is expired by the
	// sweep after 900s.
	p, clock, reg := newTestPlugin(t, `{"sessionIdleTimeoutSeconds": 7200}`)

	orphan := newRequest("orphan", map[string]string{
		"x-session-id": "s1", "user-agent": "opencode/1",
	}, 10)
	mustRequestHeader(t, p, orphan)
	mustDispatch(t, p, orphan)
	if got := gaugeValue(t, p, "opencode", "inflight"); got != 1 {
		t.Fatalf("inflight gauge before expiry = %v, want 1", got)
	}

	clock.Advance(899 * time.Second)
	p.sweepOnce(clock.Now())
	if got := counterValue(t, p, "evictions", "expired"); got != 0 {
		t.Errorf("expired before TTL = %v, want 0", got)
	}

	clock.Advance(2 * time.Second) // 901s total
	p.sweepOnce(clock.Now())
	if got := counterValue(t, p, "evictions", "expired"); got != 1 {
		t.Errorf("expired after TTL = %v, want 1", got)
	}
	if got := p.table.inflightLen(); got != 0 {
		t.Errorf("in-flight entries after expiry = %d, want 0", got)
	}
	// The recompute in the same sweep dropped the now-empty inflight
	// series; the session itself survives (idle timeout not yet reached).
	out := gatherText(t, reg)
	if strings.Contains(out, "window=inflight") {
		t.Errorf("empty inflight series should be deleted by the sweep recompute:\n%s", out)
	}
	if got := testutil.ToFloat64(p.metrics.tableSize); got != 1 {
		t.Errorf("table_size after expiry = %v, want 1", got)
	}
}

// TestRejectedBeforeDispatch covers requests that reach RequestHeader but
// never PreRequest — flow-control sheds and 429s under saturation, admission
// denials, no-endpoint and scheduler errors. They are counted and tracked as
// sessions, but never enter the in-flight gauge, and the 900s expiry
// backstop has nothing to clean up for them.
func TestRejectedBeforeDispatch(t *testing.T) {
	p, clock, _ := newTestPlugin(t, `{"sessionIdleTimeoutSeconds": 3600}`)

	for i := 0; i < 3; i++ {
		req := newRequest(fmt.Sprintf("shed-%d", i), map[string]string{
			"x-session-id": "shed-session", "user-agent": "opencode/1",
		}, 100)
		mustRequestHeader(t, p, req) // rejected before PreRequest
	}

	if got := counterValue(t, p, "requests", "opencode", "x-session-id"); got != 3 {
		t.Errorf("requests = %v, want 3", got)
	}
	if got := counterValue(t, p, "requestBytes", "opencode", "x-session-id"); got != 300 {
		t.Errorf("requestBytes = %v, want 300", got)
	}
	if got := counterValue(t, p, "started", "opencode"); got != 1 {
		t.Errorf("started = %v, want 1", got)
	}
	if got := gaugeValue(t, p, "opencode", "inflight"); got != 0 {
		t.Errorf("inflight gauge = %v, want 0", got)
	}
	if got := p.table.inflightLen(); got != 0 {
		t.Errorf("in-flight entries = %d, want 0", got)
	}

	// Long after the 900s expiry TTL, the sweep reports nothing expired:
	// no entry was ever created. The session is dropped only by the idle
	// timeout.
	clock.Advance(2 * time.Hour)
	p.sweepOnce(clock.Now())
	if got := counterValue(t, p, "evictions", "expired"); got != 0 {
		t.Errorf("expired = %v, want 0 (no entry was ever created)", got)
	}
	if got := counterValue(t, p, "evictions", "idle"); got != 1 {
		t.Errorf("idle = %v, want 1", got)
	}
}

func TestIdleEvictionAndHistograms(t *testing.T) {
	// Idle timeout well under the 900s in-flight TTL, so the "stuck"
	// session's entry is still live when the idle pass runs and the
	// in-flight protection is what keeps it.
	p, clock, _ := newTestPlugin(t, `{"sessionIdleTimeoutSeconds": 60}`)

	// Session old: two turns 600s apart, both settled, last seen 155s ago.
	old1 := newRequest("old-1", map[string]string{
		"x-session-id": "old", "user-agent": "opencode/1",
	}, 10)
	fullTurn(t, p, old1)
	clock.Advance(600 * time.Second)
	old2 := newRequest("old-2", map[string]string{
		"x-session-id": "old", "user-agent": "opencode/1",
	}, 10)
	fullTurn(t, p, old2)
	// Session stuck: one dispatched request 155s ago, never settled. Its
	// in-flight entry (155s < 900s TTL) keeps the session alive despite
	// the idle timeout having passed.
	stuck := newRequest("stuck", map[string]string{
		"x-session-id": "stuck", "user-agent": "opencode/1",
	}, 10)
	mustRequestHeader(t, p, stuck)
	mustDispatch(t, p, stuck)
	clock.Advance(145 * time.Second)
	// Session fresh: settled 10s ago.
	fresh := newRequest("fresh", map[string]string{
		"x-session-id": "fresh", "user-agent": "opencode/1",
	}, 10)
	fullTurn(t, p, fresh)
	clock.Advance(10 * time.Second)

	p.sweepOnce(clock.Now()) // old idle 155s, stuck idle 155s, fresh idle 10s

	if got := counterValue(t, p, "evictions", "idle"); got != 1 {
		t.Errorf("idle evictions = %v, want 1", got)
	}
	if got := counterValue(t, p, "evictions", "expired"); got != 0 {
		t.Errorf("expired evictions = %v, want 0 (stuck's entry is younger than the TTL)", got)
	}
	if got := testutil.ToFloat64(p.metrics.tableSize); got != 2 {
		t.Errorf("table_size after idle eviction = %v, want 2 (fresh + stuck)", got)
	}
	if got := counterValue(t, p, "started", "opencode"); got != 3 {
		t.Errorf("started = %v, want 3", got)
	}
	// The evicted session's terminal state landed in the histograms.
	if got := testutil.CollectAndCount(p.metrics.turns); got != 1 {
		t.Errorf("turns histogram observations = %d, want 1 (the evicted opencode session)", got)
	}
	if got := testutil.CollectAndCount(p.metrics.duration); got != 1 {
		t.Errorf("duration histogram observations = %d, want 1", got)
	}
	if got := counterValue(t, p, "evictions", "capacity"); got != 0 {
		t.Errorf("capacity evictions = %v, want 0", got)
	}
}

func TestCapacityEviction(t *testing.T) {
	p, clock, _ := newTestPlugin(t, `{"maxSessions": 3}`)

	// s1: idle, least recently active. s2: idle, more recent. s3: in flight.
	s1 := newRequest("r1", map[string]string{
		"x-session-id": "s1", "user-agent": "opencode/1",
	}, 10)
	fullTurn(t, p, s1)
	clock.Advance(100 * time.Second)
	s2 := newRequest("r2", map[string]string{
		"x-session-id": "s2", "user-agent": "opencode/1",
	}, 10)
	fullTurn(t, p, s2)
	clock.Advance(100 * time.Second)
	s3 := newRequest("r3", map[string]string{
		"x-session-id": "s3", "user-agent": "opencode/1",
	}, 10)
	mustRequestHeader(t, p, s3)
	mustDispatch(t, p, s3)
	clock.Advance(100 * time.Second)

	// Inserting s4 beyond the cap evicts s1 (least recently active idle
	// session), never the more recent idle s2 nor the in-flight s3.
	s4 := newRequest("r4", map[string]string{
		"x-session-id": "s4", "user-agent": "opencode/1",
	}, 10)
	mustRequestHeader(t, p, s4)
	mustDispatch(t, p, s4)

	if got := counterValue(t, p, "evictions", "capacity"); got != 1 {
		t.Errorf("capacity evictions = %v, want 1", got)
	}
	if got := p.table.len(); got != 3 {
		t.Errorf("table length = %d, want 3 (s2 + s3 + s4)", got)
	}
	// table_size is recomputed by the sweep; run one (nothing is idle past
	// the default 3600s timeout, so no further eviction happens).
	p.sweepOnce(clock.Now())
	if got := testutil.ToFloat64(p.metrics.tableSize); got != 3 {
		t.Errorf("table_size = %v, want 3", got)
	}
	if got := counterValue(t, p, "evictions", "idle"); got != 0 {
		t.Errorf("idle evictions during capacity test = %v, want 0", got)
	}
	if got := counterValue(t, p, "started", "opencode"); got != 4 {
		t.Errorf("started = %v, want 4 (s1, s2, s3, s4)", got)
	}
	// The evicted session's histograms were observed.
	if got := testutil.CollectAndCount(p.metrics.turns); got != 1 {
		t.Errorf("turns histogram observations = %d, want 1", got)
	}

	// All sessions in flight: the table may exceed the cap rather than
	// evict a live session.
	p2, _, _ := newTestPlugin(t, `{"maxSessions": 1}`)
	for _, id := range []string{"i1", "i2"} {
		req := newRequest(id, map[string]string{"x-session-id": id}, 10)
		mustRequestHeader(t, p2, req)
		mustDispatch(t, p2, req)
	}
	if got := p2.table.len(); got != 2 {
		t.Errorf("in-flight protection: table length = %d, want 2 (cap exceeded, no live session dropped)", got)
	}
	if got := counterValue(t, p2, "evictions", "capacity"); got != 0 {
		t.Errorf("capacity evictions with all in flight = %v, want 0", got)
	}
}

func TestRawSessionIDsNeverInMetrics(t *testing.T) {
	p, _, reg := newTestPlugin(t, "")

	raws := []string{"super-secret-session-alpha", "another-uuid-1234", "x" + strings.Repeat("y", 200)}
	for i, raw := range raws {
		mustRequestHeader(t, p, newRequest(fmt.Sprintf("r%d", i), map[string]string{
			"x-session-id": raw,
			"user-agent":   "opencode/1.0.2",
		}, 10))
	}
	mustRequestHeader(t, p, newRequest("none", map[string]string{"user-agent": "opencode/1.0.2"}, 10))
	p.sweepOnce(p.now())

	out := gatherText(t, reg)
	for _, raw := range raws {
		if strings.Contains(out, raw) {
			t.Errorf("raw session ID %q leaked into gathered metrics:\n%s", raw, out)
		}
		if hash := p.table.hashSession(raw); strings.Contains(out, hash) {
			t.Errorf("session hash %q leaked into gathered metrics:\n%s", hash, out)
		}
	}
	// The expected labelled series are all present.
	for _, want := range []string{
		"llm_d_epp_session_requests_total", "llm_d_epp_session_request_bytes_total",
		"llm_d_epp_sessions_started_total", "llm_d_epp_session_table_size",
		"llm_d_epp_sessions_active", "harness=opencode", "session_header=x-session-id", "session_header=none",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("gathered metrics missing %q:\n%s", want, out)
		}
	}
}

func TestPluginIsPassive(t *testing.T) {
	p, _, _ := newTestPlugin(t, "")

	headers := map[string]string{
		"x-claude-code-session-id": "session-abc",
		"user-agent":               "claude-cli/2.1.3 (external, cli)",
		"authorization":            "Bearer not-a-token",
	}
	req := newRequest("r1", headers, 42)

	mustRequestHeader(t, p, req)
	mustDispatch(t, p, req)
	settle(t, p, "r1")

	// No request attribute was written by any hook.
	if keys := req.AttributeKeys(); len(keys) != 0 {
		t.Errorf("plugin wrote request attributes %v, want none", keys)
	}
	// Headers are unchanged.
	want := map[string]string{
		"x-claude-code-session-id": "session-abc",
		"user-agent":               "claude-cli/2.1.3 (external, cli)",
		"authorization":            "Bearer not-a-token",
	}
	for k, v := range want {
		if req.Headers[k] != v {
			t.Errorf("header %q mutated: %q, want %q", k, req.Headers[k], v)
		}
	}
	if len(req.Headers) != len(want) {
		t.Errorf("header set changed: %v", req.Headers)
	}
	// FairnessID and body size untouched.
	if req.FairnessID != "" {
		t.Errorf("FairnessID modified: %q", req.FairnessID)
	}
	if req.RequestSizeBytes != 42 {
		t.Errorf("RequestSizeBytes modified: %d", req.RequestSizeBytes)
	}
	// The plugin does not declare itself a data producer.
	if _, ok := any(p).(fwkplugin.ProducerPlugin); ok {
		t.Error("plugin must not implement ProducerPlugin (it produces no data)")
	}
	// Nil requests and responses are handled without panic, including a
	// PreRequest with a nil scheduling result.
	if err := p.RequestHeader(context.Background(), nil); err != nil {
		t.Errorf("nil request: %v", err)
	}
	if err := p.PreRequest(context.Background(), nil, nil); err != nil {
		t.Errorf("nil PreRequest: %v", err)
	}
	p.ResponseBody(context.Background(), nil, nil, nil)
}

// TestMetricsRegisteredOnServedRegistry proves the collectors land on the
// registry the EPP actually serves: runner.go builds the plugin handle with
// WithMetricsRecorder(ctrlmetrics.Registry) and serves that same registry on
// :9090 /metrics. Building the plugin through a handle configured the same
// way and gathering from ctrlmetrics.Registry exercises exactly that wiring.
func TestMetricsRegisteredOnServedRegistry(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	handle := fwkplugin.NewEppHandle(ctx, nil, fwkplugin.WithMetricsRecorder(ctrlmetrics.Registry))
	pi, err := PluginFactory("served", nil, handle)
	if err != nil {
		t.Fatalf("PluginFactory: %v", err)
	}
	p := pi.(*Plugin)

	// Drive one request, one evicting sweep (so the eviction-observed
	// histograms get a series), and a second dispatched request (so the
	// sweep's recompute leaves a live sessions_active series); every
	// collector then has at least one series to expose.
	r1 := newRequest("r1", map[string]string{"x-session-id": "s1"}, 10)
	mustRequestHeader(t, p, r1)
	mustDispatch(t, p, r1)
	p.now = func() time.Time { return time.Now().Add(2 * time.Hour) }
	p.sweepOnce(p.now())
	r2 := newRequest("r2", map[string]string{"x-session-id": "s2"}, 10)
	mustRequestHeader(t, p, r2)
	mustDispatch(t, p, r2)

	families, err := ctrlmetrics.Registry.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	got := map[string]bool{}
	for _, mf := range families {
		got[mf.GetName()] = true
	}
	for _, want := range []string{
		"llm_d_epp_session_requests_total",
		"llm_d_epp_session_request_bytes_total",
		"llm_d_epp_sessions_active",
		"llm_d_epp_sessions_started_total",
		"llm_d_epp_session_turns",
		"llm_d_epp_session_duration_seconds",
		"llm_d_epp_session_table_evictions_total",
		"llm_d_epp_session_table_size",
	} {
		if !got[want] {
			t.Errorf("metric %q not present on the served registry; families: %v", want, got)
		}
	}
}

func TestSweepStopsOnContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled before the sweep starts

	p, _, _ := newTestPlugin(t, `{"gaugeRefreshSeconds": 1}`)
	done := make(chan struct{})
	go func() {
		p.runSweep(ctx, 10*time.Millisecond)
		close(done)
	}()
	select {
	case <-done:
		// The sweep returned immediately on the cancelled context.
	case <-time.After(5 * time.Second):
		t.Fatal("sweep did not stop on context cancellation")
	}
}
