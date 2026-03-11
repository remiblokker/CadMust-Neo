# CadMust-Neo

A KiCad Action Plugin that optimizes PCB component placement using simulated annealing. It minimizes total wirelength while respecting board boundaries, courtyard overlaps, and keep-out zones.

## Features

- **Simulated annealing** with auto-calibrated temperature, adaptive cooling, reheating, and greedy refinement
- **Move operators**: translate, swap, rotate, and net-aware median moves
- **Constraints**: board outline (rectangular and polygon), courtyard overlap avoidance, keep-out zones
- **Component groups**: grouped components move as rigid bodies
- **Power net detection**: automatically excludes power/ground nets from wirelength optimization
- **Net exclusion control**: choose which nets to include in optimization
- **Multi-start mode**: run multiple independent starts and keep the best result
- **Tiered settings**: Basic (one-click presets), Normal (key parameters), Expert (full control)
- **Progress display**: real-time temperature and wirelength bars with cancel support
- **Silkscreen auto-placement**: post-optimization reference designator positioning
- **Undo support**: Edit > Undo restores original placement; cancel aborts without changes
- **Zero dependencies**: pure Python, no external packages required

## Requirements

- KiCad 9.x (tested with 9.0.7)
- No additional Python packages needed

## Installation

### Manual install

1. Download or clone this repository
2. Copy (or symlink) the `plugin/` folder into your KiCad scripting plugins directory:

| OS | Plugin directory |
|----|-----------------|
| macOS | `~/Library/Preferences/kicad/9.0/scripting/plugins/CadMustNeo/` |
| Linux | `~/.local/share/kicad/9.0/scripting/plugins/CadMustNeo/` |
| Windows | `%APPDATA%\kicad\9.0\scripting\plugins\CadMustNeo\` |

3. Restart KiCad. The CadMust-Neo icon appears in the PCB Editor toolbar.

## Usage

1. Open a PCB in KiCad's PCB Editor
2. Optionally select specific components (lock any you don't want moved)
3. Click the CadMust-Neo toolbar button
4. Choose a quality preset (Fast / Balanced / Thorough) or switch to Expert mode for full control
5. Click **Optimize**
6. Review the results — accept to keep the new placement, or reject to revert

## Tips & workflow

### Recommended workflow

1. **Draw the board outline** on the Edge.Cuts layer. The optimizer needs this to keep components within bounds.
2. **Lock mechanical constraints** — connectors, mounting holes, LEDs, switches. Right-click → Properties → check "Locked".
3. **Place and lock critical ICs** (optional but recommended) — roughly position your key components (microcontrollers, FPGAs, memory ICs, etc.) and their immediate support circuitry where they need to be, then lock them. The optimizer will pull unlocked components toward these anchors via net connectivity. Alternatively, you can skip this step and let the optimizer do a first pass on everything — then lock what looks good and iterate.
4. **Position and group decoupling caps** with their IC (select all, right-click → Grouping → Group) so they stay together as a rigid body during optimization.
5. **Run the optimizer** on all remaining unlocked components to get a proper initial placement.
6. **Iterate** — if an area doesn't look right, unlock it, adjust, and re-run.

### Additional tips

**Use "Move selected only" for targeted rework.** If your board is mostly done but a specific area needs rearranging, select just those components and enable "Move selected components only". This leaves the rest of the board untouched.

**Multi-start mode prevents regression.** With multiple starts (Expert mode), start 0 always preserves your current placement as a baseline. The optimizer can only improve on it — if no better placement is found, you get your original back.

**Power and ground nets are auto-detected.** Nets like GND, VCC, VDD are automatically excluded from wirelength optimization since they'll be connected by power planes. You can override this in the net exclusion list (Normal/Expert mode).

## How it works

CadMust-Neo extracts component positions, pad locations, and net connectivity from KiCad into a pure Python model. The simulated annealing optimizer explores placement alternatives by:

1. **Calibrating** the starting temperature to achieve ~95% initial acceptance
2. **Annealing** with adaptive cooling rates based on acceptance ratio
3. **Reheating** from the best solution found so far (3 rounds)
4. **Refining** with greedy local search (translate + rotate)

The cost function combines half-perimeter wirelength (HPWL) with penalty terms for courtyard overlaps, boundary violations, and keep-out zone intrusions.

## Running tests

```bash
python3 -m unittest tests.test_optimizer -v
```

Tests use synthetic board models and don't require KiCad.

## Heritage

CadMust-Neo builds on CadMust, a RISC OS PCB CAD suite from the 1990s that included one of the earliest placement optimizers for desktop PCB design.

## License

[MIT](LICENSE)
