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
	"container/list"
	"sync"
	"time"
)

// Eviction reasons, exported as the `reason` label of
// llm_d_epp_session_table_evictions_total.
const (
	evictionReasonIdle     = "idle"
	evictionReasonCapacity = "capacity"
	evictionReasonExpired  = "expired"
)

// eviction reports one table eviction to the caller's callback. tracked
// distinguishes a session eviction (idle, capacity — the turns and duration
// histograms apply) from an expired in-flight request entry, whose session
// may still be tracked.
type eviction struct {
	reason   string
	tracked  bool
	harness  string
	turns    int64
	duration time.Duration
}

// sessionEntry is the per-session state. It is keyed in the table by the
// salted session hash; the raw session ID is never stored.
type sessionEntry struct {
	hash      string
	harness   string // fixed at the session's first request
	firstSeen time.Time
	lastSeen  time.Time
	turns     int64
	inflight  int
	elem      *list.Element // node in the LRU list; Value is the hash
}

// inflightEntry tracks one request that started but has not completed.
// sess keeps a direct pointer: if the session is evicted from the table
// while the request is in flight, the eventual decrement lands on the
// orphaned entry instead of corrupting a re-created session.
type inflightEntry struct {
	sess    *sessionEntry
	started time.Time
}

// sessionTable is the bounded table of per-session state. All methods are
// safe for concurrent use. The onEvict callback (passed to track and sweep)
// is invoked while the table lock is held, so it must not call back into the
// table.
type sessionTable struct {
	mu          sync.Mutex
	salt        []byte
	sessions    map[string]*sessionEntry
	lru         list.List // front = least recently active; Value = hash
	inflight    map[string]*inflightEntry
	maxSessions int
	idleTimeout time.Duration
	inflightTTL time.Duration
}

func newSessionTable(maxSessions int, idleTimeout, inflightTTL time.Duration) *sessionTable {
	return &sessionTable{
		salt:        newProcessSalt(),
		sessions:    make(map[string]*sessionEntry),
		inflight:    make(map[string]*inflightEntry),
		maxSessions: maxSessions,
		idleTimeout: idleTimeout,
		inflightTTL: inflightTTL,
	}
}

// track records one request for the session keyed by hash: it creates the
// session when unseen (evicting for capacity first), bumps turns and
// lastSeen, and registers the request as in-flight. It returns the session's
// fixed harness, whether the session was created by this call, and the
// session's in-flight count after the request is accounted.
func (t *sessionTable) track(hash, harness, requestID string, now time.Time, onEvict func(eviction)) (sessHarness string, created bool, inflightAfter int) {
	t.mu.Lock()
	defer t.mu.Unlock()

	e, ok := t.sessions[hash]
	if !ok {
		t.evictForCapacityLocked(now, onEvict)
		e = &sessionEntry{
			hash:      hash,
			harness:   harness,
			firstSeen: now,
			lastSeen:  now,
		}
		e.elem = t.lru.PushBack(hash)
		t.sessions[hash] = e
		created = true
	}
	e.lastSeen = now
	e.turns++
	t.lru.MoveToBack(e.elem)
	if requestID != "" {
		t.inflight[requestID] = &inflightEntry{sess: e, started: now}
		e.inflight++
	}
	return e.harness, created, e.inflight
}

// evictForCapacityLocked makes room for one new session by evicting the
// least recently active idle session. Sessions with in-flight requests are
// never evicted; when every session is in flight the table is allowed to
// exceed maxSessions rather than drop live sessions.
func (t *sessionTable) evictForCapacityLocked(now time.Time, onEvict func(eviction)) {
	if len(t.sessions) < t.maxSessions {
		return
	}
	for elem := t.lru.Front(); elem != nil; {
		hash, _ := elem.Value.(string)
		e := t.sessions[hash]
		next := elem.Next()
		if e != nil && e.inflight == 0 {
			t.removeLocked(hash, e)
			onEvict(eviction{
				reason:   evictionReasonCapacity,
				tracked:  true,
				harness:  e.harness,
				turns:    e.turns,
				duration: e.lastSeen.Sub(e.firstSeen),
			})
			return
		}
		elem = next
	}
}

// complete settles one request. It returns the session's fixed harness and
// its in-flight count after the decrement; ok is false when the request ID
// was unknown (already completed, or expired by the sweep).
func (t *sessionTable) complete(requestID string) (harness string, inflightAfter int, ok bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	ie, exists := t.inflight[requestID]
	if !exists {
		return "", 0, false
	}
	delete(t.inflight, requestID)
	if ie.sess.inflight > 0 {
		ie.sess.inflight--
	}
	return ie.sess.harness, ie.sess.inflight, true
}

// sweep expires orphaned in-flight request entries older than the TTL and
// evicts sessions idle beyond the timeout with no in-flight requests.
// Expiry runs first so a session whose last request entry just expired can
// be idle-evicted in the same pass.
func (t *sessionTable) sweep(now time.Time, onEvict func(eviction)) {
	t.mu.Lock()
	defer t.mu.Unlock()

	for id, ie := range t.inflight {
		if now.Sub(ie.started) <= t.inflightTTL {
			continue
		}
		delete(t.inflight, id)
		if ie.sess.inflight > 0 {
			ie.sess.inflight--
		}
		onEvict(eviction{reason: evictionReasonExpired})
	}

	for hash, e := range t.sessions {
		if e.inflight == 0 && now.Sub(e.lastSeen) > t.idleTimeout {
			t.removeLocked(hash, e)
			onEvict(eviction{
				reason:   evictionReasonIdle,
				tracked:  true,
				harness:  e.harness,
				turns:    e.turns,
				duration: e.lastSeen.Sub(e.firstSeen),
			})
		}
	}
}

// removeLocked drops a session from the map and the LRU list.
func (t *sessionTable) removeLocked(hash string, e *sessionEntry) {
	delete(t.sessions, hash)
	t.lru.Remove(e.elem)
}

// snapshotWindows computes the sessions_active series from the current
// table: for every harness, the number of sessions with at least one
// in-flight request (window "inflight") and the number whose lastSeen falls
// within each configured window. It also returns the table size.
func (t *sessionTable) snapshotWindows(now time.Time, windows []time.Duration) (active map[string]map[string]int, size int) {
	t.mu.Lock()
	defer t.mu.Unlock()

	active = make(map[string]map[string]int)
	ensure := func(harness string) map[string]int {
		ws, ok := active[harness]
		if !ok {
			ws = make(map[string]int, len(windows)+1)
			active[harness] = ws
		}
		return ws
	}
	for _, e := range t.sessions {
		ws := ensure(e.harness)
		if e.inflight > 0 {
			ws[windowInflight]++
		}
		for _, w := range windows {
			if now.Sub(e.lastSeen) <= w {
				ws[formatWindow(int(w.Seconds()))]++
			}
		}
	}
	return active, len(t.sessions)
}

// len returns the number of tracked sessions (for tests and diagnostics).
func (t *sessionTable) len() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.sessions)
}

// inflightLen returns the number of unsettled in-flight request entries
// (for tests).
func (t *sessionTable) inflightLen() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.inflight)
}
