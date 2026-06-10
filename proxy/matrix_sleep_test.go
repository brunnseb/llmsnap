package proxy

import (
	"net/http/httptest"
	"testing"

	"github.com/napmany/llmsnap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMatrix_WhenModelAsleep_OtherModelEvicted verifies that waking an asleep model
// evicts running models not in the same set. model1 and model2 are in separate sets,
// so waking model1 evicts model2 per normal matrix eviction rules.
func TestMatrix_WhenModelAsleep_OtherModelEvicted(t *testing.T) {
	cfg := config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
		},
		ExpandedSets: []config.ExpandedSet{
			{SetName: "set1", Models: []string{"model1"}},
			{SetName: "set2", Models: []string{"model2"}},
		},
		Matrix: &config.MatrixConfig{},
	}

	m := NewMatrix(cfg, testLogger, testLogger)
	defer m.StopProcesses(StopImmediately)

	m.processes["model1"].testHandler = newTestHandler("model1")
	m.processes["model2"].testHandler = newTestHandler("model2")

	// Load model1 first and make it ready
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	require.NoError(t, m.ProxyRequest("model1", w, req))
	assert.Equal(t, StateReady, m.processes["model1"].CurrentState())

	// Simulate model1 going to sleep (e.g., due to TTL or manual sleep)
	m.processes["model1"].forceState(StateAsleep)
	assert.Equal(t, StateAsleep, m.processes["model1"].CurrentState())

	// Load model2 (now it's running while model1 is asleep)
	req2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w2 := httptest.NewRecorder()
	require.NoError(t, m.ProxyRequest("model2", w2, req2))
	assert.Equal(t, StateReady, m.processes["model2"].CurrentState())

	// Verify running models only include model2 (model1 is asleep and shouldn't count)
	running := m.RunningModels()
	assert.Len(t, running, 1, "Only model2 should be counted as running")
	assert.Contains(t, running, "model2")
	assert.NotContains(t, running, "model1", "Asleep models should not be counted as running")

	// Now wake model1 - this SHOULD evict model2 because the fix ensures asleep
	// models don't count as "running" in the solver
	req3 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w3 := httptest.NewRecorder()
	require.NoError(t, m.ProxyRequest("model1", w3, req3))

	// model1 should be ready after waking
	assert.Equal(t, StateReady, m.processes["model1"].CurrentState())

	// model2 should have been evicted (not ready anymore)
	assert.NotEqual(t, StateReady, m.processes["model2"].CurrentState(),
		"Evicted model should no longer be running")
}

// TestMatrix_RunningModelsExcludesAsleep verifies that RunningModels() does not
// include models in asleep state. This is the core fix for the eviction bug.
func TestMatrix_RunningModelsExcludesAsleep(t *testing.T) {
	cfg := config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
		},
		ExpandedSets: []config.ExpandedSet{
			{SetName: "set1", Models: []string{"model1"}},
		},
		Matrix: &config.MatrixConfig{},
	}

	m := NewMatrix(cfg, testLogger, testLogger)
	defer m.StopProcesses(StopImmediately)

	m.processes["model1"].testHandler = newTestHandler("model1")

	// Load model1 - it should be running
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	require.NoError(t, m.ProxyRequest("model1", w, req))

	// model1 should be running
	running := m.RunningModels()
	assert.Len(t, running, 1, "model1 should be running")
	assert.Contains(t, running, "model1")

	// Put model1 to sleep (simulate)
	m.processes["model1"].forceState(StateAsleep)

	// Now no models should be in the running list
	running = m.RunningModels()
	assert.Len(t, running, 0, "No models should be counted as running when model1 is asleep")
}

// TestMatrix_WakePreservesCoLoadedModel verifies that waking an asleep model does NOT
// evict co-loaded models that belong to the same set. model1 and model2 are in the same
// set, so waking model1 should keep model2 running.
func TestMatrix_WakePreservesCoLoadedModel(t *testing.T) {
	cfg := config.Config{
		HealthCheckTimeout: 15,
		Models: map[string]config.ModelConfig{
			"model1": getTestSimpleResponderConfig("model1"),
			"model2": getTestSimpleResponderConfig("model2"),
		},
		ExpandedSets: []config.ExpandedSet{
			{SetName: "shared", Models: []string{"model1", "model2"}},
		},
		Matrix: &config.MatrixConfig{},
	}

	m := NewMatrix(cfg, testLogger, testLogger)
	defer m.StopProcesses(StopImmediately)

	m.processes["model1"].testHandler = newTestHandler("model1")
	m.processes["model2"].testHandler = newTestHandler("model2")

	// Load both models
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	require.NoError(t, m.ProxyRequest("model1", w, req))
	assert.Equal(t, StateReady, m.processes["model1"].CurrentState())

	req2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w2 := httptest.NewRecorder()
	require.NoError(t, m.ProxyRequest("model2", w2, req2))
	assert.Equal(t, StateReady, m.processes["model2"].CurrentState())

	// Put model1 to sleep
	m.processes["model1"].forceState(StateAsleep)
	assert.Equal(t, StateAsleep, m.processes["model1"].CurrentState())

	// model2 should still be running
	running := m.RunningModels()
	assert.Len(t, running, 1)
	assert.Contains(t, running, "model2")

	// Wake model1 - model2 should NOT be evicted because they're in the same set
	req3 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w3 := httptest.NewRecorder()
	require.NoError(t, m.ProxyRequest("model1", w3, req3))

	// Both models should be ready
	assert.Equal(t, StateReady, m.processes["model1"].CurrentState())
	assert.Equal(t, StateReady, m.processes["model2"].CurrentState(),
		"Co-loaded model should not be evicted when waking a model in the same set")
}
