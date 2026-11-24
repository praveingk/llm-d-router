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

package runner

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// TestSessionObserverConfigInstantiates loads the session-observer plugin
// through the production config path (parseConfigurationPhaseOne ->
// parseConfigurationPhaseTwo) with the exact parameters shipped in the
// curvebender-tools router config, and asserts the plugin instantiates and is
// registered as Alpha. The config must stay byte-identical to the deployed
// one so a parameter rename fails here instead of at rollout.
func TestSessionObserverConfigInstantiates(t *testing.T) {
	configText := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: session-observer
  name: session-observer
  parameters:
    sessionHeaders: [x-claude-code-session-id, x-opencode-session-id, x-session-id, session-id, session_id, x-session-affinity]
    harnessHeaders: [x-client-user-agent, user-agent]
    maxHarnessLabels: 50
    activeWindowsSeconds: [300, 3600]
    sessionIdleTimeoutSeconds: 3600
    maxSessions: 100000
    gaugeRefreshSeconds: 15
`

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	opts := runserver.NewOptions()
	opts.ConfigText = configText
	opts.PoolName = "session-observer-config-pool"
	opts.AllowExperimentalPlugins = true // session-observer is Alpha

	r := NewRunner()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err, "session-observer config no longer parses")

	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	_, err = r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
	require.NoError(t, err, "session-observer config no longer instantiates its plugins")

	// The plugin is registered at Alpha and present on the runner's handle.
	require.NoError(t, fwkplugin.ValidatePluginStability(r.PluginHandle, opts.AllowExperimentalPlugins))
	p := r.PluginHandle.Plugin("session-observer")
	require.NotNil(t, p, "session-observer plugin not found on the handle")
	require.Equal(t, "session-observer", p.TypedName().Type)

	// The director wires extension points by type-asserting each plugin
	// from the handle (requestcontrol.Config.AddPlugins, fed by
	// r.requestControlConfig.AddPlugins(handle.GetAllPlugins()...) in
	// parseConfigurationPhaseTwo). A plugin listed only under `plugins:` — no
	// profile stanza — reaches all three of this plugin's extension points
	// exactly when it satisfies these three interfaces, so assert them on
	// the production-config-instantiated plugin.
	_, isHeader := p.(fwkrc.RequestHeaderProcessor)
	_, isPre := p.(fwkrc.PreRequest)
	_, isBody := p.(fwkrc.ResponseBodyProcessor)
	require.True(t, isHeader, "session-observer must be collected as a RequestHeaderProcessor")
	require.True(t, isPre, "session-observer must be collected as a PreRequest plugin (in-flight accounting)")
	require.True(t, isBody, "session-observer must be collected as a ResponseBodyProcessor (completion hook)")
}
