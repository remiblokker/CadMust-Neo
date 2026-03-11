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

**Lock mechanical constraints first.** Connectors, mounting holes, LEDs, switches — anything with a fixed physical position should be locked before running the optimizer. Right-click a component → Properties → check "Locked".

**Group decoupling caps with their IC.** Rather than locking bypass capacitors in place, group them with their associated IC. They'll move together as a rigid body while the optimizer finds the best position for the pair. In KiCad, select the IC and its caps, then right-click → Grouping → Group.

**Use "Move selected only" for partial optimization.** If part of your board is already well placed, select just the components you want to rearrange and enable "Move selected components only" in the settings. The optimizer will only move the selection, leaving everything else untouched.

**Multi-start mode prevents regression.** With multiple starts (Expert mode), start 0 always preserves your current placement as a baseline. The optimizer can only improve on it — if no better placement is found, you get your original back.

**Best for initial placement.** The optimizer shines when you have a pile of unplaced components and want a good starting point. On a board that's already carefully hand-placed, there may be limited room for improvement.

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
