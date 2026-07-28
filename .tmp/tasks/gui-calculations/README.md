# GUI Calculations Section - Task Breakdown

## Overview
Add a separate "Calculations" section to the GUI that allows users to select models, configure parameters, run calculations, and view unified results.

---

## Task 01: Create Calculations Section UI Framework
**Priority**: High  
**Dependencies**: None  
**Parallel**: Yes  

### Description
Create the basic UI framework for the Calculations section with tabs for each model.

### Requirements
- Add new section/frame to main GUI (separate from current Run)
- Create tab container for model-specific tabs
- Add "Calculate" button at bottom
- Add results display area (1D plot area + 2D selection list)

### Files to Create/Modify
- `src/compton_suite/gui/calculations.py` - New file for calculations UI
- `src/compton_suite/gui/app.py` - Integrate calculations section

### Acceptance Criteria
- [ ] Calculations section appears in GUI
- [ ] Tab container with placeholders for 4 models
- [ ] Calculate button is visible but disabled until models selected
- [ ] Results area placeholder exists

---

## Task 02: Implement Per-Model Tab with Controls
**Priority**: High  
**Dependencies**: Task 01  
**Parallel**: No  

### Description
Implement per-model tabs with "Use for calculations" checkbox and model-specific parameters.

### Requirements
- For each model (KASCADE, XIGMA-I, Delta, Analytical):
  - Checkbox: "Use for calculations"
  - Parameters section (shown only when checkbox is checked)
  - Parameters match model's `extra_params()` output
- Parameters should be editable numerical fields
- Default values from model's default config

### Files to Create/Modify
- `src/compton_suite/gui/calculations.py` - Add model tab implementation

### Acceptance Criteria
- [ ] Each model has its own tab
- [ ] Checkbox toggles parameter visibility
- [ ] Parameters are editable and match model's extra_params()
- [ ] Default values are populated

---

## Task 03: Implement Multi-Model Calculation Runner
**Priority**: High  
**Dependencies**: Task 02  
**Parallel**: No  

### Description
Implement the calculation runner that runs all selected models and stores results.

### Requirements
- When "Calculate" button is clicked:
  - Collect selected models
  - Collect parameters for each selected model
  - Run each model's adapter.run() with collected parameters
  - Store results in a dictionary keyed by model name
  - Update UI with results
- Handle errors gracefully (show warning if model fails)

### Files to Create/Modify
- `src/compton_suite/gui/calculations.py` - Add calculation runner
- `src/compton_suite/gui/runner.py` - May need to extend or create new runner

### Acceptance Criteria
- [ ] Calculate button runs all selected models
- [ ] Results are stored and accessible
- [ ] Errors are caught and displayed
- [ ] UI updates with results

---

## Task 04: Implement 1D Plot Display with Legend
**Priority**: High  
**Dependencies**: Task 03  
**Parallel**: No  

### Description
Implement 1D plot display that shows results from all models on same figure with legend and different colors.

### Requirements
- Single figure for all 1D plots (spectrum, energy distribution)
- Each model gets a different color
- Legend shows which color corresponds to which model
- Plot should update when new results are available
- Handle case where model doesn't have 1D results

### Files to Create/Modify
- `src/compton_suite/gui/calculations.py` - Add 1D plot display

### Acceptance Criteria
- [ ] 1D plots from all models appear on same figure
- [ ] Different colors per model
- [ ] Legend is visible and correct
- [ ] Missing 1D results are handled gracefully

---

## Task 05: Implement 2D Plot Selection and Display
**Priority**: High  
**Dependencies**: Task 03  
**Parallel**: No  

### Description
Implement 2D plot display with a list to select and view results from a particular model.

### Requirements
- Listbox or dropdown showing models that have 2D results
- When model is selected, show its 2D plot (angular distribution)
- Plot updates when selection changes
- Handle case where model doesn't have 2D results

### Files to Create/Modify
- `src/compton_suite/gui/calculations.py` - Add 2D plot display

### Acceptance Criteria
- [ ] Selection list shows models with 2D results
- [ ] Selecting a model shows its 2D plot
- [ ] Plot updates on selection change
- [ ] Missing 2D results are handled gracefully

---

## Task 06: Integration and Testing
**Priority**: High  
**Dependencies**: Tasks 01-05  
**Parallel**: No  

### Description
Integrate all components and test the complete workflow.

### Requirements
- All components work together
- Test with multiple models selected
- Test with different parameter combinations
- Test error handling
- Test edge cases (no models selected, model fails, etc.)

### Files to Create/Modify
- `src/compton_suite/gui/calculations.py` - Final integration
- `src/compton_suite/gui/app.py` - Final integration

### Acceptance Criteria
- [ ] Complete workflow works end-to-end
- [ ] All models can be selected and run
- [ ] Results display correctly
- [ ] Error handling works

---

## Task Summary

| Task | Description | Dependencies | Parallel |
|------|-------------|--------------|----------|
| 01 | UI Framework | None | Yes |
| 02 | Per-Model Tabs | 01 | No |
| 03 | Calculation Runner | 02 | No |
| 04 | 1D Plot Display | 03 | No |
| 05 | 2D Plot Selection | 03 | No |
| 06 | Integration | 01-05 | No |

## Parallel Execution Plan
- **Batch 1**: Task 01 (UI Framework)
- **Batch 2**: Task 02 (Per-Model Tabs) - depends on 01
- **Batch 3**: Task 03 (Calculation Runner) - depends on 02
- **Batch 4**: Tasks 04 + 05 (1D + 2D plots) - both depend on 03, can run in parallel
- **Batch 5**: Task 06 (Integration) - depends on all

## Estimated Effort
- Task 01: 2-3 hours
- Task 02: 3-4 hours
- Task 03: 4-5 hours
- Task 04: 2-3 hours
- Task 05: 2-3 hours
- Task 06: 2-3 hours
- **Total**: 15-21 hours
