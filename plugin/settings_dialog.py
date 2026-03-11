"""Settings dialog with tiered complexity (Basic / Normal / Expert)."""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional, Tuple

import wx

from .annealer import SAConfig


_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

PRESETS: Dict[str, Dict[str, Any]] = {
    "Fast": {
        "max_iterations": 100,
        "cooling_fast": 0.75,
        "cooling_normal": 0.85,
        "cooling_slow": 0.92,
    },
    "Balanced": {},  # SAConfig defaults
    "Thorough": {
        "max_iterations": 500,
        "cooling_fast": 0.85,
        "cooling_normal": 0.93,
        "cooling_slow": 0.97,
    },
}

_PRESET_NAMES = ["Fast", "Balanced", "Thorough"]
_MODE_NAMES = ["Basic", "Normal", "Expert"]

_DEFAULTS: Dict[str, Any] = {
    "mode": "Basic",
    "preset": "Balanced",
    "max_iterations": 200,
    "moves_per_temp": 0,
    "calibration_samples": 200,
    "initial_accept_rate": 0.95,
    "cooling_fast": 0.80,
    "cooling_normal": 0.90,
    "cooling_slow": 0.95,
    "penalty_scale_min": 0.10,
    "num_starts": 1,
    "move_selected_only": False,
}


def load_settings() -> Dict[str, Any]:
    """Load settings from JSON file, returning defaults on any error."""
    try:
        with open(_SETTINGS_FILE, "r") as f:
            data = json.load(f)
        # Merge with defaults so new keys are always present
        merged = dict(_DEFAULTS)
        merged.update(data)
        return merged
    except Exception:
        return dict(_DEFAULTS)


def save_settings(data: Dict[str, Any]) -> None:
    """Save settings to JSON file."""
    try:
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # non-critical — settings just won't persist


class SettingsDialog(wx.Dialog):
    """CadMust-Neo settings dialog with Basic / Normal / Expert tiers.

    board_info (optional) — dict produced by cadmust_neo_action._build_board_info():
        moveable:    int   — number of moveable components
        locked:      int   — number of locked components
        signal_nets: int   — number of signal nets (non-power, >= 2 pads)
        hpwl_mm:     float — current HPWL of signal nets (mm)
        net_list:    List[Tuple[str, int, bool]]
                           — (net_name, pad_count, is_auto_power)
                             sorted by pad_count descending
                             shown as a checklist; checked = excluded from HPWL
    """

    # Minimum pad count for a signal net to appear in the exclusion list.
    _SIGNAL_NET_THRESHOLD = 8

    def __init__(self, parent,
                 board_info: Optional[Dict[str, Any]] = None):
        super().__init__(parent, title="CadMust-Neo Settings",
                         style=wx.DEFAULT_DIALOG_STYLE)

        self._settings = load_settings()
        self._board_info = board_info
        self._net_list_names: List[str] = []   # parallel to checklist items
        self._net_checklist: Optional[wx.CheckListBox] = None
        self._net_hint: Optional[wx.StaticText] = None
        self._chk_select_all: Optional[wx.CheckBox] = None
        self._net_sizer: Optional[wx.StaticBoxSizer] = None

        self._build_ui()
        self._restore_values()
        self._on_mode_change(None)

        self.SetMinSize(wx.Size(400, -1))
        self.Fit()
        self.CentreOnParent()

    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Board info (shown when board_info is provided) ---
        if self._board_info:
            bi = self._board_info
            info_box = wx.StaticBox(self, label="Board")
            info_sizer = wx.StaticBoxSizer(info_box, wx.VERTICAL)

            comp_str = f"{bi['moveable']} moveable, {bi['locked']} locked"
            net_str = f"{bi['signal_nets']} signal net{'s' if bi['signal_nets'] != 1 else ''}"
            lbl_comps = wx.StaticText(self, label=f"{comp_str}  \u00b7  {net_str}")
            info_sizer.Add(lbl_comps, 0, wx.LEFT | wx.TOP | wx.RIGHT, 6)

            n_sig = bi['signal_nets']
            avg_span = (bi['hpwl_mm'] / n_sig) if n_sig else 0.0
            hpwl_str = (f"HPWL: {bi['hpwl_mm']:,.0f} mm"
                        f"  \u00b7  avg {avg_span:.1f} mm/net")
            lbl_hpwl = wx.StaticText(self, label=hpwl_str)
            lbl_hpwl.SetForegroundColour(wx.Colour(80, 80, 80))
            lbl_hpwl.SetToolTip(
                "Half-perimeter wire length of all signal nets at current placement.\n"
                "Power nets (GND, VCC, etc.) are excluded — they connect to almost\n"
                "everything and their HPWL can't be reduced by placement.\n\n"
                "Avg mm/net = HPWL \u00f7 number of signal nets — a compact\n"
                "board typically has a lower avg span than a spread-out one."
            )
            info_sizer.Add(lbl_hpwl, 0, wx.LEFT | wx.BOTTOM | wx.RIGHT, 6)
            main_sizer.Add(info_sizer, 0, wx.ALL | wx.EXPAND, 8)

        # --- Mode selector ---
        self._mode_radio = wx.RadioBox(
            self, label="Mode", choices=_MODE_NAMES,
            majorDimension=3, style=wx.RA_SPECIFY_COLS,
        )
        self._mode_radio.SetToolTip(
            "Basic: choose a quality preset.\n"
            "Normal: also set iteration count.\n"
            "Expert: full control over all SA parameters."
        )
        self._mode_radio.Bind(wx.EVT_RADIOBOX, self._on_mode_change)
        main_sizer.Add(self._mode_radio, 0, wx.ALL | wx.EXPAND, 8)

        # --- Quality preset ---
        self._preset_radio = wx.RadioBox(
            self, label="Quality", choices=_PRESET_NAMES,
            majorDimension=3, style=wx.RA_SPECIFY_COLS,
        )
        self._preset_radio.SetToolTip(
            "Fast: 100 iterations, aggressive cooling — quick result.\n"
            "Balanced: 200 iterations — good default for most boards.\n"
            "Thorough: 500 iterations, slow cooling — best result, takes longer."
        )
        self._preset_radio.Bind(wx.EVT_RADIOBOX, self._on_preset_change)
        main_sizer.Add(self._preset_radio, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 8)

        # --- Move selected only ---
        self._chk_selected_only = wx.CheckBox(
            self, label="Move selected components only")
        self._chk_selected_only.SetToolTip(
            "When checked, only components selected in the PCB editor will be moved.\n"
            "All other components stay in place.\n"
            "Select components first in KiCad, then run the optimizer."
        )
        main_sizer.Add(self._chk_selected_only, 0,
                       wx.LEFT | wx.RIGHT | wx.TOP, 10)

        # --- Parameters box (Normal + Expert) ---
        self._params_box = wx.StaticBox(self, label="Parameters")
        params_sizer = wx.StaticBoxSizer(self._params_box, wx.VERTICAL)
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=12)
        grid.AddGrowableCol(1, 1)

        # Normal-level parameters
        lbl_iter = wx.StaticText(self, label="Max iterations:")
        lbl_iter.SetToolTip(
            "Number of temperature steps the annealer runs.\n"
            "More iterations = better result, but slower.\n"
            "Typical range: 100 (fast) to 500 (thorough)."
        )
        grid.Add(lbl_iter, 0, wx.ALIGN_CENTER_VERTICAL)
        self._spin_max_iter = wx.SpinCtrl(self, min=50, max=2000, initial=200)
        self._spin_max_iter.SetToolTip(lbl_iter.GetToolTipText())
        grid.Add(self._spin_max_iter, 0, wx.EXPAND)

        lbl_moves = wx.StaticText(self, label="Moves per temp (0=auto):")
        lbl_moves.SetToolTip(
            "How many placement moves are tried at each temperature step.\n"
            "0 = automatic (scales with number of moveable components).\n"
            "Increase for more thorough exploration at each step."
        )
        grid.Add(lbl_moves, 0, wx.ALIGN_CENTER_VERTICAL)
        self._spin_moves = wx.SpinCtrl(self, min=0, max=5000, initial=0)
        self._spin_moves.SetToolTip(lbl_moves.GetToolTipText())
        grid.Add(self._spin_moves, 0, wx.EXPAND)

        # Expert-level parameters
        self._expert_labels = []
        self._expert_ctrls = []

        def _add_expert_slider(label, val, lo, hi, step, tooltip=""):
            """Add a slider + numeric readout for a float parameter."""
            lbl = wx.StaticText(self, label=label)
            lbl.SetToolTip(tooltip)

            # Panel holds slider + value label side by side
            panel = wx.Panel(self)
            panel_sizer = wx.BoxSizer(wx.HORIZONTAL)

            multiplier = round(1.0 / step)
            slider = wx.Slider(
                panel,
                minValue=round(lo * multiplier),
                maxValue=round(hi * multiplier),
                value=round(val * multiplier),
                style=wx.SL_HORIZONTAL,
            )
            slider.SetToolTip(tooltip)

            val_label = wx.StaticText(panel, label=f"{val:.2f}",
                                      size=wx.Size(36, -1),
                                      style=wx.ALIGN_RIGHT)

            panel_sizer.Add(slider, 1, wx.ALIGN_CENTER_VERTICAL)
            panel_sizer.Add(val_label, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6)
            panel.SetSizer(panel_sizer)

            def _on_slide(evt):
                v = slider.GetValue() / multiplier
                val_label.SetLabel(f"{v:.2f}")

            slider.Bind(wx.EVT_SLIDER, _on_slide)

            # Attach GetValue/SetValue to panel so callers work unchanged
            # Use default args to capture current slider/label/multiplier
            panel.GetValue = lambda _s=slider, _m=multiplier: _s.GetValue() / _m
            panel.SetValue = lambda v, _s=slider, _l=val_label, _m=multiplier: (
                _s.SetValue(round(v * _m)),
                _l.SetLabel(f"{v:.2f}"),
            )

            grid.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(panel, 0, wx.EXPAND)
            self._expert_labels.append(lbl)
            self._expert_ctrls.append(panel)
            return panel

        self._spin_cal_samples_label = wx.StaticText(
            self, label="Calibration samples:")
        self._spin_cal_samples_label.SetToolTip(
            "Number of random moves used to estimate the starting temperature.\n"
            "Higher values give a more accurate T0 but add startup time.\n"
            "Default 200 works well for most boards."
        )
        self._spin_cal_samples = wx.SpinCtrl(self, min=50, max=1000, initial=200)
        self._spin_cal_samples.SetToolTip(
            self._spin_cal_samples_label.GetToolTipText())
        grid.Add(self._spin_cal_samples_label, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self._spin_cal_samples, 0, wx.EXPAND)
        self._expert_labels.append(self._spin_cal_samples_label)
        self._expert_ctrls.append(self._spin_cal_samples)

        self._spin_accept = _add_expert_slider(
            "Initial accept rate:", 0.95, 0.50, 0.99, 0.01,
            tooltip=(
                "Fraction of uphill moves accepted at the starting temperature.\n"
                "Higher = more exploration early on.\n"
                "0.95 is a good default (accepts 95% of bad moves initially)."
            ),
        )
        self._spin_cool_fast = _add_expert_slider(
            "Cooling (Fast):", 0.80, 0.50, 0.99, 0.01,
            tooltip=(
                "Cooling rate used when acceptance is high (> 50%).\n"
                "Temperature is multiplied by this factor each step.\n"
                "Lower = faster cooling, fewer steps, less time."
            ),
        )
        self._spin_cool_normal = _add_expert_slider(
            "Cooling (Balanced):", 0.90, 0.50, 0.99, 0.01,
            tooltip=(
                "Cooling rate used when acceptance is moderate (20–50%).\n"
                "Temperature is multiplied by this factor each step.\n"
                "0.90 gives a balanced speed/quality trade-off."
            ),
        )
        self._spin_cool_slow = _add_expert_slider(
            "Cooling (Thorough):", 0.95, 0.50, 0.99, 0.01,
            tooltip=(
                "Cooling rate used when acceptance is low (< 20%).\n"
                "Temperature is multiplied by this factor each step.\n"
                "Higher = slower cooling, more exploration, better result."
            ),
        )
        self._spin_penalty_scale = _add_expert_slider(
            "Penalty scale min:", 0.10, 0.01, 1.00, 0.05,
            tooltip=(
                "Minimum penalty scale at high temperature.\n"
                "Boundary and keepout penalties are multiplied by this\n"
                "at T=T0 and ramp up to 1.0 as temperature drops.\n"
                "Lower = more exploration (temporary violations allowed).\n"
                "Overlap is always at full weight."
            ),
        )

        self._spin_num_starts_label = wx.StaticText(
            self, label="Multi-start runs:")
        self._spin_num_starts_label.SetToolTip(
            "Number of independent SA runs.\n"
            "Run 0 starts from the current placement (no regression).\n"
            "Runs 1+ start from random positions — explores new basins.\n"
            "Total time scales linearly. Best result is kept.\n"
            "1 = single run (default)."
        )
        self._spin_num_starts = wx.SpinCtrl(self, min=1, max=20, initial=1)
        self._spin_num_starts.SetToolTip(
            self._spin_num_starts_label.GetToolTipText())
        grid.Add(self._spin_num_starts_label, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self._spin_num_starts, 0, wx.EXPAND)
        self._expert_labels.append(self._spin_num_starts_label)
        self._expert_ctrls.append(self._spin_num_starts)

        params_sizer.Add(grid, 1, wx.ALL | wx.EXPAND, 6)
        main_sizer.Add(params_sizer, 0,
                       wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, 8)

        # --- Normal-level controls (to show/hide separately from expert) ---
        self._normal_label_iter = grid.GetItem(0).GetWindow()  # "Max iterations:"
        self._normal_label_moves = grid.GetItem(2).GetWindow()  # "Moves per temp:"

        # --- Net exclusion checklist (shown when net_list is provided) ---
        net_list = (self._board_info or {}).get('net_list', [])
        if net_list:
            net_box = wx.StaticBox(self, label="HPWL net exclusion")
            self._net_sizer = wx.StaticBoxSizer(net_box, wx.VERTICAL)
            net_sizer = self._net_sizer

            hint_sizer = wx.BoxSizer(wx.HORIZONTAL)
            self._net_hint = wx.StaticText(
                self,
                label="Checked nets are excluded from wire-length optimisation:"
            )
            self._net_hint.SetForegroundColour(wx.Colour(80, 80, 80))
            self._net_hint.SetToolTip(
                "Power/ground nets (GND, VCC, +5V, …) connect to almost every\n"
                "component — their HPWL can't be reduced, so excluding them lets\n"
                "the optimizer focus on signal nets.\n\n"
                "You can also exclude high-fanout signal nets (buses, clocks)\n"
                "if they dominate the score and can't realistically be shortened."
            )
            hint_sizer.Add(self._net_hint, 1, wx.ALIGN_CENTER_VERTICAL)

            self._chk_select_all = wx.CheckBox(self, label="All")
            self._chk_select_all.SetToolTip("Check or uncheck all nets")
            self._chk_select_all.Bind(wx.EVT_CHECKBOX, self._on_select_all_nets)
            hint_sizer.Add(self._chk_select_all, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 8)

            net_sizer.Add(hint_sizer, 0, wx.ALL | wx.EXPAND, 5)

            items = [f"{name}  ({count} pads)" for name, count, _ in net_list]
            self._net_list_names = [name for name, _, _ in net_list]
            self._net_checklist = wx.CheckListBox(self, choices=items)
            # Pre-check auto-detected power nets
            for i, (_, _, is_excluded) in enumerate(net_list):
                self._net_checklist.Check(i, is_excluded)
            # Height: enough for up to ~6 items without scrolling
            item_h = max(60, min(130, len(items) * 22))
            self._net_checklist.SetMinSize(wx.Size(-1, item_h))
            net_sizer.Add(self._net_checklist, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)
            main_sizer.Add(net_sizer, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, 8)

        # --- Buttons ---
        btn_sizer = wx.StdDialogButtonSizer()
        self._btn_ok = wx.Button(self, wx.ID_OK, "Optimize")
        self._btn_ok.SetDefault()
        btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        btn_sizer.AddButton(self._btn_ok)
        btn_sizer.AddButton(btn_cancel)
        btn_sizer.Realize()
        main_sizer.Add(btn_sizer, 0,
                       wx.ALL | wx.ALIGN_RIGHT, 8)

        self._btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)

        self.SetSizer(main_sizer)

    def _restore_values(self):
        s = self._settings

        mode_idx = _MODE_NAMES.index(s.get("mode", "Basic")) \
            if s.get("mode") in _MODE_NAMES else 0
        self._mode_radio.SetSelection(mode_idx)

        preset_idx = _PRESET_NAMES.index(s.get("preset", "Balanced")) \
            if s.get("preset") in _PRESET_NAMES else 1
        self._preset_radio.SetSelection(preset_idx)

        self._spin_max_iter.SetValue(int(s.get("max_iterations", 200)))
        self._spin_moves.SetValue(int(s.get("moves_per_temp", 0)))
        self._spin_cal_samples.SetValue(int(s.get("calibration_samples", 200)))
        self._spin_accept.SetValue(float(s.get("initial_accept_rate", 0.95)))
        self._spin_cool_fast.SetValue(float(s.get("cooling_fast", 0.80)))
        self._spin_cool_normal.SetValue(float(s.get("cooling_normal", 0.90)))
        self._spin_cool_slow.SetValue(float(s.get("cooling_slow", 0.95)))
        self._spin_penalty_scale.SetValue(float(s.get("penalty_scale_min", 0.10)))
        self._spin_num_starts.SetValue(int(s.get("num_starts", 1)))
        self._chk_selected_only.SetValue(bool(s.get("move_selected_only", False)))

    def _on_preset_change(self, evt):
        """When preset changes in Expert mode, update the parameter controls."""
        mode = _MODE_NAMES[self._mode_radio.GetSelection()]
        if mode != "Expert":
            return
        preset_name = _PRESET_NAMES[self._preset_radio.GetSelection()]
        defaults = SAConfig()
        overrides = PRESETS.get(preset_name, {})
        for key, val in overrides.items():
            setattr(defaults, key, val)
        self._spin_max_iter.SetValue(defaults.max_iterations)
        self._spin_cool_fast.SetValue(defaults.cooling_fast)
        self._spin_cool_normal.SetValue(defaults.cooling_normal)
        self._spin_cool_slow.SetValue(defaults.cooling_slow)

    def _on_mode_change(self, evt):
        mode = _MODE_NAMES[self._mode_radio.GetSelection()]

        show_preset = True  # Quality preset visible in all modes
        show_params = mode in ("Normal", "Expert")
        show_expert = mode == "Expert"

        self._preset_radio.Show(show_preset)
        self._params_box.Show(show_params)

        # Normal-level controls
        self._normal_label_iter.Show(show_params)
        self._spin_max_iter.Show(show_params)
        self._normal_label_moves.Show(show_params)
        self._spin_moves.Show(show_params)

        # Expert-level controls
        for lbl in self._expert_labels:
            lbl.Show(show_expert)
        for ctrl in self._expert_ctrls:
            ctrl.Show(show_expert)

        # Net exclusion — Normal + Expert only
        if self._net_sizer is not None:
            self._net_sizer.GetStaticBox().Show(show_params)
            self._net_hint.Show(show_params)
            self._chk_select_all.Show(show_params)
            self._net_checklist.Show(show_params)

        self.Layout()
        self.Fit()

    @property
    def move_selected_only(self) -> bool:
        """True if only selected components should be moved."""
        return self._chk_selected_only.GetValue()

    @property
    def excluded_net_names(self):
        """Set of net names the user has ticked for HPWL exclusion."""
        if not self._net_list_names or self._net_checklist is None:
            return set()
        return {
            self._net_list_names[i]
            for i in range(len(self._net_list_names))
            if self._net_checklist.IsChecked(i)
        }

    def _on_select_all_nets(self, evt):
        """Check or uncheck all nets in the exclusion list."""
        check = self._chk_select_all.GetValue()
        if self._net_checklist is not None:
            for i in range(self._net_checklist.GetCount()):
                self._net_checklist.Check(i, check)

    def build_config(self) -> SAConfig:
        """Build an SAConfig from current dialog state."""
        mode = _MODE_NAMES[self._mode_radio.GetSelection()]
        config = SAConfig()

        # Apply preset as baseline (all modes)
        preset_name = _PRESET_NAMES[self._preset_radio.GetSelection()]
        overrides = PRESETS.get(preset_name, {})
        for key, val in overrides.items():
            setattr(config, key, val)

        if mode == "Normal":
            config.max_iterations = self._spin_max_iter.GetValue()
            config.moves_per_temp = self._spin_moves.GetValue()
        elif mode == "Expert":
            # Expert controls override preset values
            config.max_iterations = self._spin_max_iter.GetValue()
            config.moves_per_temp = self._spin_moves.GetValue()
            config.calibration_samples = self._spin_cal_samples.GetValue()
            config.initial_accept_rate = self._spin_accept.GetValue()
            config.cooling_fast = self._spin_cool_fast.GetValue()
            config.cooling_normal = self._spin_cool_normal.GetValue()
            config.cooling_slow = self._spin_cool_slow.GetValue()
            config.penalty_scale_min = self._spin_penalty_scale.GetValue()
            config.num_starts = self._spin_num_starts.GetValue()
        return config

    def _on_ok(self, evt):
        # Save current values for next session
        data = {
            "mode": _MODE_NAMES[self._mode_radio.GetSelection()],
            "preset": _PRESET_NAMES[self._preset_radio.GetSelection()],
            "max_iterations": self._spin_max_iter.GetValue(),
            "moves_per_temp": self._spin_moves.GetValue(),
            "calibration_samples": self._spin_cal_samples.GetValue(),
            "initial_accept_rate": self._spin_accept.GetValue(),
            "cooling_fast": self._spin_cool_fast.GetValue(),
            "cooling_normal": self._spin_cool_normal.GetValue(),
            "cooling_slow": self._spin_cool_slow.GetValue(),
            "penalty_scale_min": self._spin_penalty_scale.GetValue(),
            "num_starts": self._spin_num_starts.GetValue(),
            "move_selected_only": self._chk_selected_only.GetValue(),
        }
        save_settings(data)
        self.EndModal(wx.ID_OK)
