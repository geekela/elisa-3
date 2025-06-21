# Guide: Testing What Happens After 3 Seconds Without Input

## Overview
Your supervisor asked: "What does the bump do after 3s? Does it drift (good) or disappear (bad)?"

This guide explains how to run the tests I've created to answer this question.

## Setup Instructions

1. **First, ensure you have a trained model**
   - Run cells 1-7 in `enhanced_training_demo.ipynb` to train a model
   - This creates the `trained_model` variable needed for testing

2. **Run the focused 3-second test (Cell 22)**
   - This provides a single detailed analysis
   - Shows exactly what happens at the 3-second mark
   - Generates suggested text  

3. **Run statistical analysis (Cell 23)**
   - Runs 20 trials for confidence intervals
   - Provides mean ± SD values for your methods section
   - Classifies behavior as drift, mixed, or decay

## Understanding the Results

### Good Outcome (Drift)
- Bump maintains >50% amplitude at 3s
- Shows gradual drift (< 30°/s)
- Similar to biological HD cells in darkness
- **Report this as**: "The bump exhibits gradual drift similar to biological HD cells"

### Poor Outcome (Disappearance)
- Bump amplitude < 30% at 3s
- Rapid decay indicates a limitation
- **Report this as**: "We observed persistence of the bump following input removal, but it is short-lived"

### Mixed Outcome
- Between 30-50% amplitude retention
- Shows both drift and decay
- **Report this as**: "The bump shows variable persistence, representing a limitation requiring future optimization"

## Key Metrics to Report

1. **Amplitude at 3s**: X% ± Y% of initial (from Cell 23)
2. **Drift at 3s**: X° ± Y° (from Cell 23)
3. **Drift rate**: X°/s ± Y°/s (from Cell 23)
4. **Persistence time**: How long until amplitude < 20% of initial

## Biological Comparison
- Real HD cells drift at 5-15°/s in darkness (Valerio & Taube, 2016)
- Your model should ideally fall within this range

## Visualizations  

The tests generate several figures:
1. **Cell 22 Figure**: Shows bump evolution with 3s marker - good for main results
2. **Cell 23 Statistics**: Shows distributions - good for supplementary material
3. **Neural Activity Snapshots**: Shows bump shape at different times

## Quick Decision Tree

```
If amplitude at 3s > 50% AND drift rate < 30°/s:
    → Report as DRIFT (good, like real brains)
    → Emphasize biological similarity
    
Elif amplitude at 3s > 30%:
    → Report as MIXED behavior
    → Mention as current limitation
    
Else:
    → Report as DECAY (limitation)
    → Mention need for future work
```

## Example Report Text

Based on your results, Cell 22 will generate appropriate text. Here's an example:

**For drift behavior (good):**
> "Short-term memory: The bump persists for >3s without input, exhibiting gradual drift at 12.3°/s rather than disappearing. At 3 seconds, the bump maintains 68% of its initial amplitude while drifting by only 15.2°. This drift behavior is consistent with recordings from rodent head direction cells in darkness (Valerio & Taube, 2016), where neurons show similar drift rates of 5-15°/s."

**For decay behavior (limitation):**
> "Short-term memory: We observed persistence of the bump following input removal, but it is short-lived. The bump amplitude decays to 25% of its initial value within 3 seconds. This rapid decay represents a limitation of the current model that should be addressed in future work." 