# Head Direction Ring Attractor Model - Research Guide

## Overview

This document outlines the research project on modeling head direction (HD) cells in the Anterodorsal Nucleus (ADN) of the thalamus using a ring attractor model. The ultimate goal is to adapt existing models to predict behaviors observed through optogenetic experiments.

## Project Evolution Timeline

### Initial Phase - Foundational Understanding

**Core Objective**: Build a simple model that can be later integrated with optogenetic behavioral observations.

**Key Concepts Established**:
- HD cells are primarily found in ADN (a small region with poor viral uptake for optogenetics)
- Ring attractor dynamics are specific to this region (with some activity in pre/postSubiculum)
- Focus on rate-based models rather than spiking models

### Phase 1 - Basic Model Architecture

**Initial 4-Step Plan**:
1. **Build neuron model**: Rate-based model (not spiking)
2. **Build ring structure**: Neurons arranged in a ring topology
3. **Test width tuning**: Match ADN width characteristics
4. **Add refinements**: Account for post-stimulation inertia effects

**Input Types for Each Neuron**:
1. Real light (optogenetic stimulation)
2. Cue input (sensory/vestibular)
3. Neighboring cell connections (recurrent network)

### Phase 2 - Simplification and Focus

**Key Decisions**:
- **Do NOT use Brian2** simulation framework
- Keep model as simple as possible initially
- Exclude gain factors, plasticity, or visual cues in early versions
- Focus on connectivity matrix definition
- Treat mammillary bodies as external input source
- Eventually, might need to add Angular Head velocity cells
- Primary input: vestibular (external to ADN but intrinsic to mouse, coming from vestibular organs)

### Phase 3 - Critical Model Revisions

**Major Changes**:
1. **No pre-initialized bump** - Activity bump must emerge from external inputs
2. **Fix weight matrix issues** - Identified errors in initial implementation
3. **Update to recent literature** - Move beyond memory-based models

**Model Architecture Refinements**:
- Ring of N neurons (0-360° preferred directions)
- Local excitation with Gaussian weight profile
- No autapses (zero diagonal in weight matrix)
- Global inhibition from extrinsic sources
- Continuous-time rate dynamics with tanh nonlinearity, but other nonlinear functions can be explored

### Phase 4 - Testing and Validation Framework

**Test Scenarios Developed**:
1. **Basic Tracking Tests**:
   - Sudden turns (0→180° in 1 second)
   - Sequential movements (left 90°→0°→right 90°→0°)
   - Complex sequences (left 20°→right 90°→left 30°→right 60°)
   - Extreme velocity: 180°/s turning speed

2. **Advanced Tests**:
   - Noise resistance
   - Cue conflict resolution
   - Cue integration (weighted combination)
   - Latency analysis as function of velocity

### Phase 5 - Model Integration and Biology

**Key Biological Constraints**:
- HD cells are **excitatory only** (no Gaussian inhibitors)
- Global long-range inhibition instead of local
- Inhibitory neurons receive tonic activation
- Random connectivity from inhibitory to excitatory neurons
- Reciprocal connections (W_EI and W_IE) with different weights

**Network Components**:
- **W_EE**: Excitatory-to-excitatory connections
- **W_EI**: Excitatory-to-inhibitory connections
- **W_IE**: Inhibitory-to-excitatory connections
- **W_II**: Inhibitory-to-inhibitory (later removed)

### Phase 6 - Current Implementation Requirements

**Final Model Structure**:
1. **Connectivity**:
   - W_EI should be random (not binary)
   - W_IE and W_EI should look similar in structure
   - Remove W_II connections
   - Add Poisson noise to individual neurons

2. **Visualization Requirements**:
   - Weight matrices as CSV files
   - Heatmaps with consistent color scaling (1.0 weight = green)
   - 3D parameter space visualization:
     - X-axis: Number of connected neurons
     - Y-axis: W_IE weight values
     - Z-axis: Excitatory firing rate

3. **Parameter Optimization**:
   - Implement network learning
   - Test multiple search strategies:
     - Bayesian Optimization
     - Adam Optimization
     - Random Search
     - Grid Search
   - Robustness analysis across parameter ranges

## Implementation Checklist

### Core Model Components
- [ ] Rate-based neural dynamics with tanh activation
- [ ] Ring topology with N neurons (0-360° coverage)
- [ ] Gaussian local excitation profile
- [ ] Global inhibition mechanism
- [ ] No autapses (zero diagonal weights)
- [ ] Vestibular input integration
- [ ] Poisson noise on individual neurons

### Testing Suite
- [ ] Bump formation from noise
- [ ] Tracking under various movement patterns
- [ ] Extreme velocity handling (180°/s)
- [ ] Persistence after input removal
- [ ] Cue conflict scenarios
- [ ] Latency measurements
- [ ] Parameter robustness analysis

### Visualization Tools
- [ ] Activity bump profiles
- [ ] Decoded position time series
- [ ] Weight matrix heatmaps
- [ ] 3D parameter space plots
- [ ] Ring-view animations

### Documentation
- [ ] Clear parameter definitions (kappa, init_stim, etc.)
- [ ] Inhibition mechanism explanation
- [ ] Results for initialization at 0°
- [ ] Firing rate limits analysis

## Key Technical Decisions

1. **Rate Model vs Spiking**: Use rate-based for simplicity and interpretability
2. **Inhibition Type**: Global long-range, not local Gaussian
3. **Input Sources**: Vestibular as primary, optogenetic for future validation
4. **Initialization**: Allow natural bump formation, not pre-initialized
5. **Noise Model**: Poisson noise on individual neurons
6. **Parameter Search**: Multiple optimization techniques for robustness

## Success Criteria

1. **Model Performance**:
   - Stable bump formation and maintenance
   - Accurate tracking under various velocities
   - Robust to noise perturbations
   - Biologically plausible dynamics

2. **Technical Requirements**:
   - Clean, well-commented code
   - Comprehensive test suite
   - Publication-quality visualizations
   - Detailed parameter analysis

3. **Documentation**:
   - Complete internship report
   - Presentation-ready materials
   - Clear explanation of model choices
   - Comparison with literature

## Future Extensions

Once the core model is validated:
1. Integrate optogenetic stimulation patterns
2. Compare model predictions with experimental data
3. Explore plasticity mechanisms
4. Add visual cue integration
5. Investigate gain modulation effects

## References to Consult

- Touretzky & Redish book chapter (Fig 18.x)
- Jayaraman et al., 2020 (Drosophila HD system)
- Compte et al., 2000 (Working memory dynamics)
- Peyrache et al., 2015 (ADN recordings)
- Ajabi et al., 2023 (Mammalian attractor circuits)
